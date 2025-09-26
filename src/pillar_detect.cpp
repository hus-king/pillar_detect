#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/transforms.h>
#include <tf/transform_listener.h>
#include <Eigen/Dense>
#include <locale.h>
#include <deque>
#include <mutex>
#include <map>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

// 全局柱子结构体
struct GlobalPillar {
    Eigen::Vector2f xy;         // XY平面位置
    float radius;               // 半径
    float z_min, z_max;         // Z方向范围
    ros::Time last_seen;        // 最后一次观测时间
    int observation_count;      // 被观测到的次数（用于置信度）
    bool confirmed = false;     // 是否已确认为真实東子
    
    GlobalPillar() : xy(0, 0), radius(0), z_min(0), z_max(0), observation_count(0) {}
};

// 柱子候选结构体
struct PillarCandidate {
    float x, y;                 // XY中心位置
    float radius;               // 半径
    float z_min, z_max;         // Z范围
    int point_count;            // 点数
    
    PillarCandidate() : x(0), y(0), radius(0), z_min(0), z_max(0), point_count(0) {}
};

class PillarDetector {
public:
    PillarDetector(ros::NodeHandle& nh) : nh_(nh), processing_count_(0) {
        // 设置本地化，确保中文正确显示
        setlocale(LC_ALL, "zh_CN.UTF-8");
        
        // 从参数服务器读取参数
        // 点云积累参数
        nh_.param<int>("max_accumulated_clouds", max_accumulated_clouds_, 5);
        nh_.param<double>("accumulation_voxel_size", accumulation_voxel_size_, 0.05);
        nh_.param<bool>("enable_accumulation", enable_accumulation_, true);
        nh_.param<double>("processing_timeout", processing_timeout_, 0.1); // 默认100ms超时
        
        // 点云预处理参数
        nh_.param<double>("voxel_size", voxel_size_, 0.05);
        nh_.param<double>("height_min", height_min_, 0.5);
        nh_.param<double>("height_max", height_max_, 3.0);
        
        // 聚类参数
        nh_.param<double>("cluster_tolerance", cluster_tolerance_, 0.2);
        nh_.param<int>("min_cluster_size", min_cluster_size_, 20);
        nh_.param<int>("max_cluster_size", max_cluster_size_, 10000);
        
        // 柱子检测参数
        nh_.param<double>("min_pillar_height", min_pillar_height_, 0.5);
        nh_.param<double>("max_pillar_radius", max_pillar_radius_, 0.08);
        
        // 柱子保留和自适应采样参数
        nh_.param<bool>("enable_pillar_preservation", enable_pillar_preservation_, true);
        nh_.param<bool>("enable_adaptive_sampling", enable_adaptive_sampling_, true);
        nh_.param<bool>("enable_connect_broken_pillars", enable_connect_broken_pillars_, true);
        nh_.param<float>("pillar_density_factor", pillar_density_factor_, 0.6);
        nh_.param<int>("pillar_enhancement_points", pillar_enhancement_points_, 4);
        nh_.param<float>("fine_voxel_factor", fine_voxel_factor_, 0.5);
        nh_.param<float>("pillar_connection_xy_threshold", pillar_connection_xy_threshold_, 0.2);
        nh_.param<float>("pillar_connection_max_z_gap", pillar_connection_max_z_gap_, 0.8);
        nh_.param<float>("z_gap_fill_threshold", z_gap_fill_threshold_, 0.1);
        
        // 统计输出参数
        nh_.param<bool>("enable_point_count_output", enable_point_count_output_, true);
        nh_.param<int>("detailed_stats_interval", detailed_stats_interval_, 10);
        nh_.param<bool>("enable_detailed_pillar_info", enable_detailed_pillar_info_, true);
        
        // 混合检测模式参数
        nh_.param<bool>("enable_hybrid_detection", enable_hybrid_detection_, true);
        nh_.param<int>("min_observation_to_confirm", min_observation_to_confirm_, 2);
        nh_.param<float>("pillar_merge_distance", pillar_merge_distance_, 0.3);
        nh_.param<bool>("enable_exploration", enable_exploration_, true);
        nh_.param<float>("known_region_expansion", known_region_expansion_, 0.5);
        nh_.param<float>("pillar_z_search_margin", pillar_z_search_margin_, 1.0);
        nh_.param<float>("new_pillar_height_factor", new_pillar_height_factor_, 0.9);
        
        // 话题配置
        std::string input_topic, pillar_topic, accumulated_topic;
        nh_.param<std::string>("input_cloud_topic", input_topic, "/cloud_registered");
        nh_.param<std::string>("pillar_cloud_topic", pillar_topic, "/cloud_pillar");
        nh_.param<std::string>("accumulated_cloud_topic", accumulated_topic, "/cloud_accumulated");
        
        // 订阅输入点云
        sub_cloud_ = nh_.subscribe(input_topic, 1, &PillarDetector::cloudCallback, this);
        // 发布检测到的杆子点云
        pub_pillar_ = nh_.advertise<sensor_msgs::PointCloud2>(pillar_topic, 1);
        // 发布积累的点云
        pub_accumulated_ = nh_.advertise<sensor_msgs::PointCloud2>(accumulated_topic, 1);
        
        // 初始化全局柱子地图相关
        pillar_centroids_cloud_.reset(new PointCloudT);
        pillar_kdtree_.setInputCloud(pillar_centroids_cloud_);
        next_pillar_id_ = 1;

        ROS_INFO("柱子检测器已初始化");
        ROS_INFO("订阅话题: %s", input_topic.c_str());
        ROS_INFO("发布话题: %s, %s", pillar_topic.c_str(), accumulated_topic.c_str());
        ROS_INFO("积累点云设置: 最大积累帧数=%d, 降采样体素大小=%.3fm, 积累功能=%s", 
                max_accumulated_clouds_, accumulation_voxel_size_, enable_accumulation_ ? "启用" : "禁用");
        ROS_INFO("柱子检测参数: 最小高度=%.2fm, 最大半径=%.2fm", 
                min_pillar_height_, max_pillar_radius_);
        ROS_INFO("柱子保留设置: 启用=%s, 自适应采样=%s, 精细体素因子=%.2f", 
                enable_pillar_preservation_ ? "是" : "否", 
                enable_adaptive_sampling_ ? "是" : "否", 
                fine_voxel_factor_);
        ROS_INFO("断开柱子连接设置: 启用=%s, XY阈值=%.2fm, 最大Z间隙=%.2fm", 
                enable_connect_broken_pillars_ ? "是" : "否",
                pillar_connection_xy_threshold_,
                pillar_connection_max_z_gap_);
        ROS_INFO("统计输出设置: 启用=%s, 详细统计间隔=%d次, 详细柱子信息=%s", 
                enable_point_count_output_ ? "是" : "否",
                detailed_stats_interval_,
                enable_detailed_pillar_info_ ? "是" : "否");
        ROS_INFO("混合检测模式: 启用=%s, 探索新柱子=%s, 确认观测次数=%d", 
                enable_hybrid_detection_ ? "是" : "否",
                enable_exploration_ ? "是" : "否",
                min_observation_to_confirm_);
        ROS_INFO("高度动态更新: Z搜索边界=%.2fm, 新柱子高度因子=%.2f", 
                pillar_z_search_margin_,
                new_pillar_height_factor_);
    }

private:
    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        try {
            // 记录开始处理时间，用于性能监控
            ros::WallTime start_time = ros::WallTime::now();
            
            // 转换为 PCL 点云
            PointCloudT::Ptr cloud(new PointCloudT);
            pcl::fromROSMsg(*msg, *cloud);

            if (cloud->empty()) {
                ROS_WARN("接收到空点云，跳过处理");
                return;
            }

            // 记录点云帧ID和大小，便于调试
            static int frame_count = 0;
            int current_frame = ++frame_count;
            ROS_DEBUG("处理点云帧 #%d，包含 %zu 个点", current_frame, cloud->size());

        // 1. 体素滤波降采样（可选，加速处理）
        PointCloudT::Ptr cloud_filtered(new PointCloudT);
        pcl::VoxelGrid<PointT> voxel;
        voxel.setInputCloud(cloud);
        voxel.setLeafSize(voxel_size_, voxel_size_, voxel_size_); // 使用参数配置的体素大小
        voxel.filter(*cloud_filtered);

        // 2. 高度滤波：只保留一定高度范围内的点
        PointCloudT::Ptr cloud_height(new PointCloudT);
        pcl::PassThrough<PointT> pass;
        pass.setInputCloud(cloud_filtered);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(height_min_, height_max_); // 使用参数配置的高度范围
        pass.filter(*cloud_height);

        if (cloud_height->empty()) {
            ROS_WARN("高度滤波后点云为空，跳过处理");
            return;
        }
        
        // 将高度过滤后的点云添加到积累队列
        addCloudToAccumulation(cloud_height, msg->header);
        
        // 获取积累的点云进行处理
        PointCloudT::Ptr processing_cloud;
        if (enable_accumulation_) {
            processing_cloud = getAccumulatedCloud();
            ROS_DEBUG("使用积累的点云进行处理，包含 %zu 个点", processing_cloud->size());
        } else {
            processing_cloud = cloud_height;
            ROS_DEBUG("高度滤波后剩余 %zu 个点", cloud_height->size());
        }

        // 使用混合检测模式
        if (enable_hybrid_detection_) {
            // 混合检测流程
            hybridPillarDetection(processing_cloud, msg->header);
        } else {
            // 传统全图检测模式
            traditionalPillarDetection(processing_cloud, msg->header);
        }
        
        // 计算并记录处理时间，监控性能
        ros::WallTime end_time = ros::WallTime::now();
        double execution_time = (end_time - start_time).toSec() * 1000; // 毫秒
        ROS_DEBUG("点云处理耗时: %.2f ms", execution_time);
        
        // 如果处理时间过长，发出警告
        if (execution_time > 80.0) { // 超过80ms（假设10Hz发布频率）可能会导致处理跟不上
            ROS_WARN("点云处理时间较长(%.2f ms)，可能无法跟上数据发布速度", execution_time);
        }
        } catch (const std::exception& e) {
            ROS_ERROR("点云处理过程中发生异常: %s", e.what());
        } catch (...) {
            ROS_ERROR("点云处理过程中发生未知异常");
        }
    }
    
    // 混合检测模式：结合全局柱子地图和新区域探索
    void hybridPillarDetection(const PointCloudT::Ptr& cloud, const std_msgs::Header& header) {
        ros::WallTime start_time = ros::WallTime::now();
        
        // 1. 构建当前帧的"未知区域"点云
        PointCloudT::Ptr unknown_region_cloud = extractUnknownRegion(cloud);
        
        // 2. 对未知区域运行轻量级新柱子检测
        std::vector<PillarCandidate> new_candidates;
        if (enable_exploration_ && !unknown_region_cloud->empty()) {
            new_candidates = detectNewPillars(unknown_region_cloud);
            ROS_DEBUG("在未知区域检测到 %zu 个新柱子候选", new_candidates.size());
        }
        
        // 3. 将新候选加入全局地图（去重、融合）
        if (!new_candidates.empty()) {
            updateGlobalPillarMap(new_candidates, header.stamp);
        }
        
        // 4. 对所有已确认的柱子做快速验证和提取
        PointCloudT::Ptr final_pillar_cloud = validateAndExtractPillars(cloud);
        
        // 5. 发布结果
        publishPillarResult(final_pillar_cloud, header);
        
        // 性能统计
        double execution_time = (ros::WallTime::now() - start_time).toSec() * 1000;
        ROS_INFO("混合检测完成: 全局柱子%zu个, 已确认%zu个, 本次检测%zu个点, 耗时%.2fms", 
                 global_pillar_map_.size(), getConfirmedPillarCount(), 
                 final_pillar_cloud->size(), execution_time);
    }
    
    // 传统全图检测模式（保持原有逻辑）
    void traditionalPillarDetection(const PointCloudT::Ptr& cloud, const std_msgs::Header& header) {
        // 3. 欧氏聚类（基于距离的聚类）
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
        tree->setInputCloud(cloud);

        pcl::EuclideanClusterExtraction<PointT> ec;
        ec.setClusterTolerance(cluster_tolerance_); // 使用参数配置的聚类距离
        ec.setMinClusterSize(min_cluster_size_);   // 使用参数配置的最小聚类点数
        ec.setMaxClusterSize(max_cluster_size_);   // 使用参数配置的最大聚类点数
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(cluster_indices);

        ROS_INFO("检测到 %zu 个聚类簇", cluster_indices.size());

        // 4. 筛选"杆子"候选簇
        PointCloudT::Ptr pillar_cloud(new PointCloudT);
        int pillar_count = 0;
        for (const auto& indices : cluster_indices) {
            PointCloudT::Ptr cluster(new PointCloudT);
            pcl::copyPointCloud(*cloud, indices, *cluster);
            
            // 计算包围盒或主方向
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(*cluster, centroid);
            
            // 计算Z方向的范围和最高点
            float min_z = std::numeric_limits<float>::max();
            float max_z = -std::numeric_limits<float>::max();
            for (const auto& pt : cluster->points) {
                if (pt.z < min_z) min_z = pt.z;
                if (pt.z > max_z) max_z = pt.z;
            }
            float height = max_z - min_z;
            
            // 计算XY平面投影的几何特征
            float max_radius = 0.0f;
            float min_x = std::numeric_limits<float>::max();
            float max_x = -std::numeric_limits<float>::max();
            float min_y = std::numeric_limits<float>::max();
            float max_y = -std::numeric_limits<float>::max();
            
            for (const auto& pt : cluster->points) {
                // 计算半径
                float dx = pt.x - centroid[0];
                float dy = pt.y - centroid[1];
                float r = std::sqrt(dx*dx + dy*dy);
                if (r > max_radius) max_radius = r;
                
                // 计算XY包围盒用于宽度计算
                if (pt.x < min_x) min_x = pt.x;
                if (pt.x > max_x) max_x = pt.x;
                if (pt.y < min_y) min_y = pt.y;
                if (pt.y > max_y) max_y = pt.y;
            }
            
            // 计算宽度（包围盒的对角线长度）
            float width_x = max_x - min_x;
            float width_y = max_y - min_y;
            float bbox_width = std::sqrt(width_x*width_x + width_y*width_y);
            float diameter = max_radius * 2.0f; // 直径
            
            // 启发式判断：高而细（使用配置的参数）
            if (height > min_pillar_height_ && max_radius < max_pillar_radius_) {
                // 可选：检查是否"孤立"——周围一定范围内无其他大簇
                // 此处简化：仅用几何特征
                pillar_count++;
                
                // 输出详细的柱子信息（如果启用）
                if (enable_detailed_pillar_info_) {
                    ROS_INFO("=== 检测到柱子 #%d ===", pillar_count);
                    ROS_INFO("  位置中心: (%.2f, %.2f, %.2f)", centroid[0], centroid[1], centroid[2]);
                    ROS_INFO("  高度: %.3fm (从 %.3fm 到 %.3fm)", height, min_z, max_z);
                    ROS_INFO("  最高点高度: %.3fm", max_z);
                    ROS_INFO("  宽度信息:");
                    ROS_INFO("    - 最大半径: %.3fm", max_radius);
                    ROS_INFO("    - 直径: %.3fm", diameter);
                    ROS_INFO("    - 包围盒宽度: %.3fm (X:%.3f, Y:%.3f)", bbox_width, width_x, width_y);
                    ROS_INFO("  点数量: %zu 个点", cluster->size());
                    ROS_INFO("  点密度: %.1f 点/m³", (float)cluster->size() / (M_PI * max_radius * max_radius * height));
                    ROS_INFO("  高宽比: %.2f", height / (diameter > 0 ? diameter : 0.001));
                    ROS_INFO("========================");
                } else {
                    ROS_INFO("检测到柱子 %d: 高度=%.3fm, 最大半径=%.3fm, 最高点=%.3fm, 点数=%zu", 
                             pillar_count, height, max_radius, max_z, cluster->size());
                }
                
                *pillar_cloud += *cluster;
            }
        }

        // 5. 发布结果
        publishPillarResult(pillar_cloud, header);
        ROS_INFO("传统检测完成，共找到 %d 个柱子，发布 %zu 个点", pillar_count, pillar_cloud->size());
    }
    
    // 发布柱子检测结果
    void publishPillarResult(const PointCloudT::Ptr& pillar_cloud, const std_msgs::Header& header) {
        sensor_msgs::PointCloud2 output_msg;
        pcl::toROSMsg(*pillar_cloud, output_msg);
        output_msg.header = header; // 保持时间戳和坐标系
        pub_pillar_.publish(output_msg);
    }

    // 将当前点云添加到积累队列中
    void addCloudToAccumulation(const PointCloudT::Ptr& cloud, const std_msgs::Header& header) {
        if (!enable_accumulation_) return;
        
        try {
            std::lock_guard<std::mutex> lock(cloud_mutex_);
            
            // 深拷贝点云以避免潜在的共享内存问题
            PointCloudT::Ptr cloud_copy(new PointCloudT);
            pcl::copyPointCloud(*cloud, *cloud_copy);
            
            // 添加点云到队列
            accumulated_clouds_.push_back(cloud_copy);
            latest_header_ = header;
            
            // 如果超过最大点云数量，移除最旧的
            while (accumulated_clouds_.size() > max_accumulated_clouds_) {
                accumulated_clouds_.pop_front();
            }
            
            // 发布积累的点云（但不在互斥锁中调用getAccumulatedCloud()来避免死锁）
            PointCloudT::Ptr accumulated = getAccumulatedCloudLocked();
            if (!accumulated->empty()) {
                // 统计积累前的原始点数
                size_t total_raw_points = 0;
                for (const auto& cloud : accumulated_clouds_) {
                    total_raw_points += cloud->size();
                }
                
                // 发布积累的点云
                sensor_msgs::PointCloud2 output_msg;
                pcl::toROSMsg(*accumulated, output_msg);
                output_msg.header = latest_header_;
                pub_accumulated_.publish(output_msg);
                
                // 实时输出详细统计信息（如果启用）
                if (enable_point_count_output_) {
                    static int publish_count = 0;
                    publish_count++;
                    
                    ROS_INFO("=== 积累点云统计 (第%d次发布) ===", publish_count);
                    ROS_INFO("原始点云帧数: %zu", accumulated_clouds_.size());
                    ROS_INFO("原始总点数: %zu", total_raw_points);
                    ROS_INFO("降采样后点数: %zu", accumulated->size());
                    ROS_INFO("压缩率: %.2f%% (保留了%.1f%%的点)", 
                             (1.0 - (double)accumulated->size() / total_raw_points) * 100.0,
                             (double)accumulated->size() / total_raw_points * 100.0);
                    ROS_INFO("平均每帧点数: %.1f", 
                             accumulated_clouds_.empty() ? 0.0 : (double)total_raw_points / accumulated_clouds_.size());
                    ROS_INFO("================================");
                    
                    // 根据配置的间隔输出更详细的信息
                    if (detailed_stats_interval_ > 0 && publish_count % detailed_stats_interval_ == 0) {
                        ROS_INFO(">>> 每%d次发布统计摘要 <<<", detailed_stats_interval_);
                        ROS_INFO("当前配置 - 启用积累:%s, 柱子保留:%s, 自适应采样:%s, 断开连接:%s", 
                                 enable_accumulation_ ? "是" : "否",
                                 enable_pillar_preservation_ ? "是" : "否", 
                                 enable_adaptive_sampling_ ? "是" : "否",
                                 enable_connect_broken_pillars_ ? "是" : "否");
                        ROS_INFO("体素大小: %.3fm, 最大帧数: %d", 
                                 accumulation_voxel_size_, max_accumulated_clouds_);
                    }
                } else {
                    // 简化输出，只显示基本信息
                    ROS_INFO("发布积累点云: %zu个点 (来自%zu帧)", accumulated->size(), accumulated_clouds_.size());
                }
            } else {
                ROS_WARN("积累点云为空，跳过发布");
            }
        } catch (const std::exception& e) {
            ROS_ERROR("添加点云到积累队列时出错: %s", e.what());
        }
    }
    
    // 优化点云以保留柱状结构
    PointCloudT::Ptr preservePillars(const PointCloudT::Ptr& cloud) {
        // 如果不启用柱子保留优化，直接返回原点云
        if (!enable_pillar_preservation_) return cloud;
        
        PointCloudT::Ptr result(new PointCloudT);
        *result = *cloud; // 先复制原始点云
        
        try {
            // 对点云进行聚类，找出潜在的柱子
            pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
            tree->setInputCloud(cloud);
            
            std::vector<pcl::PointIndices> cluster_indices;
            pcl::EuclideanClusterExtraction<PointT> ec;
            ec.setClusterTolerance(cluster_tolerance_ * 0.8);  // 使用更小的聚类距离以捕获细柱子
            ec.setMinClusterSize(min_cluster_size_ / 2);       // 使用更小的簇大小以捕获细柱子
            ec.setMaxClusterSize(max_cluster_size_);
            ec.setSearchMethod(tree);
            ec.setInputCloud(cloud);
            ec.extract(cluster_indices);
            
            // 储存潜在柱子簇，用于后续连接断开部分
            std::vector<PointCloudT::Ptr> potential_pillars;
            std::vector<Eigen::Vector4f> pillar_centroids;
            
            // 遍历所有聚类簇，找出潜在柱子并加强其信号
            for (const auto& indices : cluster_indices) {
                PointCloudT::Ptr cluster(new PointCloudT);
                pcl::copyPointCloud(*cloud, indices, *cluster);
                
                // 计算簇的属性
                Eigen::Vector4f centroid;
                pcl::compute3DCentroid(*cluster, centroid);
                
                // 计算Z方向的范围
                float min_z = std::numeric_limits<float>::max();
                float max_z = -std::numeric_limits<float>::max();
                for (const auto& pt : cluster->points) {
                    if (pt.z < min_z) min_z = pt.z;
                    if (pt.z > max_z) max_z = pt.z;
                }
                float height = max_z - min_z;
                
                // 计算XY平面投影的半径
                float max_radius = 0.0f;
                for (const auto& pt : cluster->points) {
                    float dx = pt.x - centroid[0];
                    float dy = pt.y - centroid[1];
                    float r = std::sqrt(dx*dx + dy*dy);
                    if (r > max_radius) max_radius = r;
                }
                
                // 如果簇可能是柱子（高而细），则增强它并记录下来
                if (height > min_pillar_height_ * 0.3 && max_radius < max_pillar_radius_ * 1.2) {
                    // 记录潜在柱子簇和其质心，用于后续连接处理
                    potential_pillars.push_back(cluster);
                    pillar_centroids.push_back(centroid);
                }
            }
            
            // 查找并连接可能是同一柱子但被分开的部分
            if (enable_connect_broken_pillars_ && potential_pillars.size() > 1) {
                connectBrokenPillars(potential_pillars, pillar_centroids, result);
            } else {
                // 如果不进行断开柱子连接，则直接增强各个柱子片段
                for (size_t i = 0; i < potential_pillars.size(); ++i) {
                    // 为细柱子添加额外点以防止它们在降采样中消失
                    PointCloudT::Ptr enhanced_cluster(new PointCloudT);
                    enhanceThinPillar(potential_pillars[i], enhanced_cluster, pillar_centroids[i]);
                    
                    // 将增强后的柱子点云添加到结果中
                    *result += *enhanced_cluster;
                }
            }
            
            return result;
        } catch (const std::exception& e) {
            ROS_WARN("保留柱子优化过程中出错: %s", e.what());
            return cloud;  // 如果处理失败，返回原始点云
        }
    }
    
    // 增强细柱子，添加额外点以防止在降采样时消失
    void enhanceThinPillar(const PointCloudT::Ptr& pillar, PointCloudT::Ptr& enhanced_pillar, const Eigen::Vector4f& centroid) {
        *enhanced_pillar = *pillar;  // 先复制原始点云
        
        // 找出z轴的最小和最大值
        float min_z = std::numeric_limits<float>::max();
        float max_z = -std::numeric_limits<float>::max();
        for (const auto& pt : pillar->points) {
            if (pt.z < min_z) min_z = pt.z;
            if (pt.z > max_z) max_z = pt.z;
        }
        
        // 如果柱子足够高，但z方向的点数不足，则添加中间点
        float height = max_z - min_z;
        if (height > min_pillar_height_ * 0.7) {
            // 获取柱子在z轴上的点密度
            std::map<int, bool> z_levels;
            float z_resolution = accumulation_voxel_size_ * 0.5; // 使用更精细的z方向分辨率
            
            for (const auto& pt : pillar->points) {
                int z_level = static_cast<int>((pt.z - min_z) / z_resolution);
                z_levels[z_level] = true;
            }
            
            // 如果密度不足，添加缺失的z层
            int expected_levels = static_cast<int>(height / z_resolution) + 1;
            if (z_levels.size() < expected_levels * pillar_density_factor_) {
                float x_center = centroid[0];
                float y_center = centroid[1];
                
                // 计算柱子的XY平面半径
                float radius = 0.0;
                for (const auto& pt : pillar->points) {
                    float dx = pt.x - x_center;
                    float dy = pt.y - y_center;
                    float r = std::sqrt(dx*dx + dy*dy);
                    if (r > radius) radius = r;
                }
                
                // 添加额外点以填充缺失的z层
                for (int i = 0; i <= expected_levels; i++) {
                    if (z_levels.find(i) == z_levels.end()) {
                        // 这个高度层没有点，添加几个
                        float z = min_z + i * z_resolution;
                        
                        // 在柱子周围添加几个点
                        for (int j = 0; j < pillar_enhancement_points_; j++) {
                            // 使用固定角度而不是随机，确保稳定的增强效果
                            float angle = j * (2.0 * M_PI / pillar_enhancement_points_);
                            float r = radius * 0.8; // 稍微向内
                            
                            PointT new_pt;
                            new_pt.x = x_center + r * cos(angle);
                            new_pt.y = y_center + r * sin(angle);
                            new_pt.z = z;
                            enhanced_pillar->points.push_back(new_pt);
                        }
                    }
                }
                
                // 更新点云大小
                enhanced_pillar->width = enhanced_pillar->points.size();
                enhanced_pillar->height = 1;
                enhanced_pillar->is_dense = false;
            }
        }
    }
    
    // 获取积累的点云（已持有锁的版本）
    PointCloudT::Ptr getAccumulatedCloudLocked() {
        // 注意：调用此函数前必须已持有cloud_mutex_锁
        PointCloudT::Ptr result(new PointCloudT);
        if (accumulated_clouds_.empty()) return result;
        
        // 合并所有点云，同时优先保留可能的柱子
        for (const auto& cloud : accumulated_clouds_) {
            // 如果启用了柱子保留优化，先处理点云以增强细柱子
            if (enable_pillar_preservation_) {
                PointCloudT::Ptr preserved = preservePillars(cloud);
                *result += *preserved;
            } else {
                *result += *cloud;
            }
        }
        
        // 降采样积累的点云
        if (result->size() > 0) {
            try {
                PointCloudT::Ptr downsampled(new PointCloudT);
                
                if (enable_adaptive_sampling_) {
                    // 使用自适应体素大小降采样
                    adaptiveVoxelDownsample(result, downsampled);
                } else {
                    // 使用常规体素降采样
                    pcl::VoxelGrid<PointT> voxel;
                    voxel.setInputCloud(result);
                    voxel.setLeafSize(accumulation_voxel_size_, accumulation_voxel_size_, accumulation_voxel_size_);
                    voxel.filter(*downsampled);
                }
                
                return downsampled;
            } catch (const std::exception& e) {
                ROS_ERROR("降采样积累点云时出错: %s", e.what());
                return result; // 返回未降采样的结果
            }
        }
        
        return result;
    }
    
    // 线程安全地获取积累的点云
    PointCloudT::Ptr getAccumulatedCloud() {
        try {
            std::lock_guard<std::mutex> lock(cloud_mutex_);
            return getAccumulatedCloudLocked();
        } catch (const std::exception& e) {
            ROS_ERROR("获取积累点云时出错: %s", e.what());
            return PointCloudT::Ptr(new PointCloudT);
        }
    }
    
    ros::NodeHandle nh_;
    ros::Subscriber sub_cloud_;
    ros::Publisher pub_pillar_;
    ros::Publisher pub_accumulated_;
    
    // 使用自适应体素大小进行降采样，对细结构使用更小的体素
    void adaptiveVoxelDownsample(const PointCloudT::Ptr& input_cloud, PointCloudT::Ptr& output_cloud) {
        // 如果输入为空，直接返回空结果
        if (input_cloud->empty()) {
            output_cloud->clear();
            return;
        }
        
        // 首先使用较大的体素尺寸对整个点云进行降采样
        PointCloudT::Ptr coarse_cloud(new PointCloudT);
        pcl::VoxelGrid<PointT> coarse_voxel;
        coarse_voxel.setInputCloud(input_cloud);
        coarse_voxel.setLeafSize(accumulation_voxel_size_, accumulation_voxel_size_, accumulation_voxel_size_);
        coarse_voxel.filter(*coarse_cloud);
        
        // 然后尝试识别可能的细柱子区域
        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
        tree->setInputCloud(coarse_cloud);
        
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<PointT> ec;
        ec.setClusterTolerance(cluster_tolerance_ * 0.8);
        ec.setMinClusterSize(min_cluster_size_ / 2);
        ec.setMaxClusterSize(max_cluster_size_);
        ec.setSearchMethod(tree);
        ec.setInputCloud(coarse_cloud);
        ec.extract(cluster_indices);
        
        // 创建一个输出点云
        *output_cloud = *coarse_cloud;
        
        // 遍历簇，找出可能的柱子，对它们使用更精细的体素大小
        for (const auto& indices : cluster_indices) {
            PointCloudT::Ptr cluster(new PointCloudT);
            pcl::copyPointCloud(*coarse_cloud, indices, *cluster);
            
            // 计算簇的属性
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(*cluster, centroid);
            
            // 计算Z方向范围和XY平面半径
            float min_z = std::numeric_limits<float>::max();
            float max_z = -std::numeric_limits<float>::max();
            float max_radius = 0.0f;
            
            for (const auto& pt : cluster->points) {
                if (pt.z < min_z) min_z = pt.z;
                if (pt.z > max_z) max_z = pt.z;
                
                float dx = pt.x - centroid[0];
                float dy = pt.y - centroid[1];
                float r = std::sqrt(dx*dx + dy*dy);
                if (r > max_radius) max_radius = r;
            }
            
            float height = max_z - min_z;
            
            // 如果簇可能是柱子(高而细)，则使用更精细的降采样
            if (height > min_pillar_height_ * 0.7 && max_radius < max_pillar_radius_ * 1.2) {
                // 找出原始点云中该区域的所有点
                PointCloudT::Ptr potential_pillar(new PointCloudT);
                
                for (const auto& pt : input_cloud->points) {
                    float dx = pt.x - centroid[0];
                    float dy = pt.y - centroid[1];
                    float dz = pt.z - (min_z + height/2);  // 到柱子中心的Z距离
                    
                    float xy_dist = std::sqrt(dx*dx + dy*dy);
                    
                    // 如果点在柱子区域内
                    if (xy_dist < max_radius * 1.5 && std::abs(dz) < height/2 * 1.2) {
                        potential_pillar->points.push_back(pt);
                    }
                }
                
                if (!potential_pillar->empty()) {
                    potential_pillar->width = potential_pillar->points.size();
                    potential_pillar->height = 1;
                    potential_pillar->is_dense = false;
                    
                    // 使用更精细的体素尺寸降采样
                    PointCloudT::Ptr fine_sampled(new PointCloudT);
                    pcl::VoxelGrid<PointT> fine_voxel;
                    fine_voxel.setInputCloud(potential_pillar);
                    
                    // 对高度方向使用更精细的分辨率
                    float fine_leaf = accumulation_voxel_size_ * fine_voxel_factor_;
                    fine_voxel.setLeafSize(fine_leaf, fine_leaf, fine_leaf * 0.7);
                    fine_voxel.filter(*fine_sampled);
                    
                    // 将精细降采样的柱子点云添加到结果中
                    *output_cloud += *fine_sampled;
                }
            }
        }
    }
    
    // 提取"未知区域"点云：排除已知柱子附近的点
    PointCloudT::Ptr extractUnknownRegion(const PointCloudT::Ptr& cloud) {
        PointCloudT::Ptr unknown(new PointCloudT);
        
        // 如果还没有已确认的柱子，整帧都是未知区域
        if (getConfirmedPillarCount() == 0) {
            *unknown = *cloud;
            return unknown;
        }

        // 使用KDTree快速查询每个点是否靠近已知柱子
        for (const auto& pt : cloud->points) {
            std::vector<int> indices;
            std::vector<float> sqr_distances;
            
            pcl::PointXYZ search_pt(pt.x, pt.y, 0); // 只关心XY平面
            float search_radius = max_pillar_radius_ + known_region_expansion_;
            
            if (pillar_kdtree_.radiusSearch(search_pt, search_radius, indices, sqr_distances) > 0) {
                // 该点在某个已知柱子附近 → 跳过（属于已知区域）
                continue;
            }
            unknown->points.push_back(pt);
        }
        
        unknown->width = unknown->points.size();
        unknown->height = 1;
        unknown->is_dense = false;
        
        ROS_DEBUG("未知区域点数: %zu / %zu (%.1f%%)", 
                  unknown->size(), cloud->size(), 
                  cloud->empty() ? 0.0 : (double)unknown->size() / cloud->size() * 100.0);
        
        return unknown;
    }
    
    // 轻量级新柱子检测：仅对未知区域进行检测
    std::vector<PillarCandidate> detectNewPillars(const PointCloudT::Ptr& cloud) {
        std::vector<PillarCandidate> candidates;
        if (cloud->empty()) return candidates;

        // 轻量聚类（稍微放宽参数，避免过度碎片化）
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
        tree->setInputCloud(cloud);

        pcl::EuclideanClusterExtraction<PointT> ec;
        ec.setClusterTolerance(cluster_tolerance_ * 1.2);  // 稍大一点，避免碎片
        ec.setMinClusterSize(min_cluster_size_ / 2);       // 更小簇也接受
        ec.setMaxClusterSize(max_cluster_size_);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(cluster_indices);

        int candidate_count = 0;
        for (const auto& indices : cluster_indices) {
            PointCloudT::Ptr cluster(new PointCloudT);
            pcl::copyPointCloud(*cloud, indices, *cluster);
            
            // 计算几何特征
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(*cluster, centroid);
            
            // 计算Z方向范围和XY几何信息
            float min_z = std::numeric_limits<float>::max();
            float max_z = -std::numeric_limits<float>::max();
            float max_radius = 0.0f;
            float min_x = std::numeric_limits<float>::max();
            float max_x = -std::numeric_limits<float>::max();
            float min_y = std::numeric_limits<float>::max();
            float max_y = -std::numeric_limits<float>::max();
            
            for (const auto& pt : cluster->points) {
                if (pt.z < min_z) min_z = pt.z;
                if (pt.z > max_z) max_z = pt.z;
                
                float dx = pt.x - centroid[0];
                float dy = pt.y - centroid[1];
                float r = std::sqrt(dx*dx + dy*dy);
                if (r > max_radius) max_radius = r;
                
                // XY包围盒
                if (pt.x < min_x) min_x = pt.x;
                if (pt.x > max_x) max_x = pt.x;
                if (pt.y < min_y) min_y = pt.y;
                if (pt.y > max_y) max_y = pt.y;
            }
            
            float height = max_z - min_z;
            float width_x = max_x - min_x;
            float width_y = max_y - min_y;
            float bbox_width = std::sqrt(width_x*width_x + width_y*width_y);
            float diameter = max_radius * 2.0f;
            
            // 判断是否像柱子：使用更严格的高度标准，减少噪声
            if (height > min_pillar_height_ * new_pillar_height_factor_ && max_radius < max_pillar_radius_ * 1.2) {
                PillarCandidate candidate;
                candidate.x = centroid[0];
                candidate.y = centroid[1];
                candidate.radius = max_radius;
                candidate.z_min = min_z;
                candidate.z_max = max_z;
                candidate.point_count = cluster->size();
                candidates.push_back(candidate);
                
                candidate_count++;
                
                // 输出详细的新柱子候选信息（如果启用）
                if (enable_detailed_pillar_info_) {
                    ROS_INFO("=== 新柱子候选 #%d ===", candidate_count);
                    ROS_INFO("  位置中心: (%.2f, %.2f, %.2f)", centroid[0], centroid[1], centroid[2]);
                    ROS_INFO("  高度: %.3fm (从 %.3fm 到 %.3fm)", height, min_z, max_z);
                    ROS_INFO("  最高点高度: %.3fm", max_z);
                    ROS_INFO("  宽度信息:");
                    ROS_INFO("    - 最大半径: %.3fm", max_radius);
                    ROS_INFO("    - 直径: %.3fm", diameter);
                    ROS_INFO("    - 包围盒宽度: %.3fm (X:%.3f, Y:%.3f)", bbox_width, width_x, width_y);
                    ROS_INFO("  点数量: %zu 个点", cluster->size());
                    ROS_INFO("  点密度: %.1f 点/m³", (float)cluster->size() / (M_PI * max_radius * max_radius * height));
                    ROS_INFO("  高宽比: %.2f", height / (diameter > 0 ? diameter : 0.001));
                    ROS_INFO("  [状态: 新发现，等待确认]");
                    ROS_INFO("========================");
                } else {
                    ROS_INFO("新柱子候选 #%d: 高度=%.3fm, 最大半径=%.3fm, 最高点=%.3fm, 点数=%zu [待确认]", 
                             candidate_count, height, max_radius, max_z, cluster->size());
                }
            }
        }
        
        if (candidates.empty()) {
            ROS_DEBUG("在未知区域中未发现新的柱子候选");
        } else {
            ROS_INFO("在未知区域中发现 %zu 个新柱子候选", candidates.size());
        }
        
        return candidates;
    }
    
    // 更新全局柱子地图：融合新候选，去重，确认
    void updateGlobalPillarMap(const std::vector<PillarCandidate>& new_candidates, const ros::Time& now) {
        std::lock_guard<std::mutex> lock(pillar_map_mutex_);
        
        for (const auto& cand : new_candidates) {
            bool merged = false;
            
            // 检查是否与已有柱子重合
            for (auto& [id, pillar] : global_pillar_map_) {
                float dist = (pillar.xy - Eigen::Vector2f(cand.x, cand.y)).norm();
                if (dist < pillar_merge_distance_) {
                    // 同一根柱子：融合数据
                    float weight = (float)pillar.observation_count / (pillar.observation_count + 1);
                    pillar.xy = pillar.xy * weight + Eigen::Vector2f(cand.x, cand.y) * (1.0f - weight);
                    pillar.radius = std::min(std::max(pillar.radius, cand.radius), (float)max_pillar_radius_);
                    pillar.z_min = std::min(pillar.z_min, cand.z_min);
                    pillar.z_max = std::max(pillar.z_max, cand.z_max);
                    pillar.last_seen = now;
                    pillar.observation_count++;
                    
                    // 判断是否可以确认为真实柱子
                    if (pillar.observation_count >= min_observation_to_confirm_) {
                        pillar.confirmed = true;
                    }
                    
                    merged = true;
                    ROS_DEBUG("合并柱子候选到ID %d, 观测次数: %d", id, pillar.observation_count);
                    break;
                }
            }
            
            if (!merged) {
                // 新柱子：添加到地图
                GlobalPillar new_pillar;
                new_pillar.xy = Eigen::Vector2f(cand.x, cand.y);
                new_pillar.radius = std::min(cand.radius, (float)max_pillar_radius_);
                new_pillar.z_min = cand.z_min;
                new_pillar.z_max = cand.z_max;
                new_pillar.last_seen = now;
                new_pillar.observation_count = 1;
                new_pillar.confirmed = false; // 需要多次观测确认

                int new_id = next_pillar_id_++;
                global_pillar_map_[new_id] = new_pillar;
                
                ROS_INFO("发现新柱子候选 ID %d: (%.2f, %.2f), 高度=%.2fm, 半径=%.2fm", 
                         new_id, cand.x, cand.y, cand.z_max - cand.z_min, cand.radius);
            }
        }

        // 更新KDTree用于下一次查询
        updatePillarKdTree();
    }
    
    // 验证和提取所有已确认柱子的点云
    PointCloudT::Ptr validateAndExtractPillars(const PointCloudT::Ptr& cloud) {
        PointCloudT::Ptr result(new PointCloudT);
        std::lock_guard<std::mutex> lock(pillar_map_mutex_);
        
        int validated_pillar_count = 0;
        
        // 遍历所有已确认的柱子
        for (const auto& [id, pillar] : global_pillar_map_) {
            if (!pillar.confirmed) continue;
            
            // 提取该柱子附近的点 - 使用扩展的Z搜索范围
            PointCloudT::Ptr pillar_points(new PointCloudT);
            for (const auto& pt : cloud->points) {
                float dx = pt.x - pillar.xy[0];
                float dy = pt.y - pillar.xy[1];
                float xy_dist = std::sqrt(dx*dx + dy*dy);
                
                // 检查是否在柱子的XY范围内和扩展的Z范围内
                if (xy_dist <= pillar.radius * 1.5 && 
                    pt.z >= pillar.z_min - pillar_z_search_margin_ && 
                    pt.z <= pillar.z_max + pillar_z_search_margin_) {
                    pillar_points->points.push_back(pt);
                }
            }
            
            if (!pillar_points->empty()) {
                pillar_points->width = pillar_points->points.size();
                pillar_points->height = 1;
                pillar_points->is_dense = false;
                
                // 计算当前帧中该柱子的详细统计信息
                Eigen::Vector4f centroid;
                pcl::compute3DCentroid(*pillar_points, centroid);
                
                float min_z = std::numeric_limits<float>::max();
                float max_z = -std::numeric_limits<float>::max();
                float max_radius_current = 0.0f;
                float min_x = std::numeric_limits<float>::max();
                float max_x = -std::numeric_limits<float>::max();
                float min_y = std::numeric_limits<float>::max();
                float max_y = -std::numeric_limits<float>::max();
                
                for (const auto& pt : pillar_points->points) {
                    // Z范围
                    if (pt.z < min_z) min_z = pt.z;
                    if (pt.z > max_z) max_z = pt.z;
                    
                    // 当前帧中的半径
                    float dx = pt.x - centroid[0];
                    float dy = pt.y - centroid[1];
                    float r = std::sqrt(dx*dx + dy*dy);
                    if (r > max_radius_current) max_radius_current = r;
                    
                    // XY包围盒
                    if (pt.x < min_x) min_x = pt.x;
                    if (pt.x > max_x) max_x = pt.x;
                    if (pt.y < min_y) min_y = pt.y;
                    if (pt.y > max_y) max_y = pt.y;
                }
                
                float height = max_z - min_z;
                float width_x = max_x - min_x;
                float width_y = max_y - min_y;
                float bbox_width = std::sqrt(width_x*width_x + width_y*width_y);
                float diameter = max_radius_current * 2.0f;
                
                // ✅ 关键：动态更新全局柱子的高度范围和其他属性
                auto& mutable_pillar = const_cast<GlobalPillar&>(pillar);
                bool height_updated = false;
                bool radius_updated = false;
                
                // 更新Z范围
                if (min_z < mutable_pillar.z_min) {
                    mutable_pillar.z_min = min_z;
                    height_updated = true;
                }
                if (max_z > mutable_pillar.z_max) {
                    mutable_pillar.z_max = max_z;
                    height_updated = true;
                }
                
                // 更新半径（如果当前观测到更大的半径，但不超过最大半径限制）
                if (max_radius_current > mutable_pillar.radius) {
                    mutable_pillar.radius = std::min(max_radius_current, (float)max_pillar_radius_);
                    radius_updated = true;
                }
                
                // 更新位置（加权平均，给更多观测更高权重）
                float weight = 1.0f / (mutable_pillar.observation_count + 1.0f);
                mutable_pillar.xy = mutable_pillar.xy * (1.0f - weight) + 
                                   Eigen::Vector2f(centroid[0], centroid[1]) * weight;
                
                // 更新观测统计
                mutable_pillar.last_seen = ros::Time::now();
                mutable_pillar.observation_count++;
                
                validated_pillar_count++;
                
                // 输出详细的柱子验证信息（如果启用）
                if (enable_detailed_pillar_info_) {
                    ROS_INFO("=== 已知柱子 #%d (ID: %d) 验证 %s ===", 
                             validated_pillar_count, id,
                             (height_updated || radius_updated) ? "[已更新]" : "");
                    ROS_INFO("  历史信息:");
                    ROS_INFO("    - 全局位置: (%.2f, %.2f)", mutable_pillar.xy[0], mutable_pillar.xy[1]);
                    ROS_INFO("    - 历史观测次数: %d", mutable_pillar.observation_count);
                    ROS_INFO("    - 全局半径: %.3fm %s", mutable_pillar.radius, radius_updated ? "[已更新]" : "");
                    ROS_INFO("    - 全局Z范围: %.3f ~ %.3fm %s", mutable_pillar.z_min, mutable_pillar.z_max, height_updated ? "[已扩展]" : "");
                    ROS_INFO("  当前帧信息:");
                    ROS_INFO("    - 当前位置: (%.2f, %.2f, %.2f)", centroid[0], centroid[1], centroid[2]);
                    ROS_INFO("    - 当前高度: %.3fm (从 %.3fm 到 %.3fm)", height, min_z, max_z);
                    ROS_INFO("    - 最高点高度: %.3fm", max_z);
                    ROS_INFO("    - 当前宽度信息:");
                    ROS_INFO("      * 最大半径: %.3fm", max_radius_current);
                    ROS_INFO("      * 直径: %.3fm", diameter);
                    ROS_INFO("      * 包围盒宽度: %.3fm (X:%.3f, Y:%.3f)", bbox_width, width_x, width_y);
                    ROS_INFO("    - 当前点数量: %zu 个点", pillar_points->size());
                    if (height > 0 && diameter > 0) {
                        ROS_INFO("    - 当前点密度: %.1f 点/m³", (float)pillar_points->size() / (M_PI * max_radius_current * max_radius_current * height));
                        ROS_INFO("    - 当前高宽比: %.2f", height / diameter);
                    }
                    
                    // 显示更新状态
                    if (height_updated) {
                        ROS_INFO("    🔺 柱子高度范围已扩展！全局高度现在为 %.3fm", mutable_pillar.z_max - mutable_pillar.z_min);
                    }
                    if (radius_updated) {
                        ROS_INFO("    📏 柱子半径已更新为 %.3fm", mutable_pillar.radius);
                    }
                    ROS_INFO("================================");
                } else {
                    ROS_INFO("已知柱子 #%d (ID: %d): 高度=%.3fm, 最大半径=%.3fm, 最高点=%.3fm, 点数=%zu, 观测%d次", 
                             validated_pillar_count, id, height, max_radius_current, max_z, pillar_points->size(), pillar.observation_count);
                }
                
                *result += *pillar_points;
            } else {
                ROS_WARN("柱子 ID %d 在当前帧中未找到点云数据", id);
            }
        }
        
        ROS_INFO("混合检测模式：验证了 %d 个已知柱子", validated_pillar_count);
        return result;
    }
        
    // 更新柱子质心的KDTree，用于快速区域查询
    void updatePillarKdTree() {
        pillar_centroids_cloud_->clear();
        
        for (const auto& [id, pillar] : global_pillar_map_) {
            if (!pillar.confirmed) continue;  // 只包含已确认的柱子
            
            pcl::PointXYZ pt;
            pt.x = pillar.xy[0];
            pt.y = pillar.xy[1];
            pt.z = 0;  // KDTree查询时只关心XY平面
            pillar_centroids_cloud_->points.push_back(pt);
        }
        
        pillar_centroids_cloud_->width = pillar_centroids_cloud_->points.size();
        pillar_centroids_cloud_->height = 1;
        pillar_centroids_cloud_->is_dense = true;
        
        if (!pillar_centroids_cloud_->empty()) {
            pillar_kdtree_.setInputCloud(pillar_centroids_cloud_);
        }
    }
    
    // 获取已确认柱子的数量
    size_t getConfirmedPillarCount() const {
        std::lock_guard<std::mutex> lock(pillar_map_mutex_);
        size_t count = 0;
        for (const auto& [id, pillar] : global_pillar_map_) {
            if (pillar.confirmed) count++;
        }
        return count;
    }
    
    // 连接因雷达扫描缺失而断开的柱子
    void connectBrokenPillars(const std::vector<PointCloudT::Ptr>& potential_pillars, 
                              const std::vector<Eigen::Vector4f>& pillar_centroids,
                              PointCloudT::Ptr& result) {
        if (potential_pillars.empty()) return;
        
        // 储存已连接的柱子片段索引
        std::vector<bool> processed(potential_pillars.size(), false);
        
        // 遍历所有潜在柱子片段
        for (size_t i = 0; i < potential_pillars.size(); ++i) {
            if (processed[i]) continue;  // 跳过已处理的片段
            
            // 计算当前片段的高度范围
            float min_z_i = std::numeric_limits<float>::max();
            float max_z_i = -std::numeric_limits<float>::max();
            for (const auto& pt : potential_pillars[i]->points) {
                if (pt.z < min_z_i) min_z_i = pt.z;
                if (pt.z > max_z_i) max_z_i = pt.z;
            }
            
            // 创建一个完整柱子点云，初始包含当前片段
            PointCloudT::Ptr complete_pillar(new PointCloudT);
            *complete_pillar = *potential_pillars[i];
            processed[i] = true;
            
            // XY位置
            float x_center = pillar_centroids[i][0];
            float y_center = pillar_centroids[i][1];
            
            // 搜索可能属于同一柱子的其他片段
            bool found_connection = true;
            while (found_connection) {
                found_connection = false;
                
                for (size_t j = 0; j < potential_pillars.size(); ++j) {
                    if (processed[j]) continue;  // 跳过已处理的片段
                    
                    float x_j = pillar_centroids[j][0];
                    float y_j = pillar_centroids[j][1];
                    
                    // 计算XY平面上的距离
                    float xy_dist = std::sqrt(
                        (x_j - x_center) * (x_j - x_center) + 
                        (y_j - y_center) * (y_j - y_center)
                    );
                    
                    // 计算片段j的高度范围
                    float min_z_j = std::numeric_limits<float>::max();
                    float max_z_j = -std::numeric_limits<float>::max();
                    for (const auto& pt : potential_pillars[j]->points) {
                        if (pt.z < min_z_j) min_z_j = pt.z;
                        if (pt.z > max_z_j) max_z_j = pt.z;
                    }
                    
                    // 判断是否可能是同一柱子的断开部分
                    // 条件：XY位置接近，Z方向上有一定的间隔但不要太远
                    if (xy_dist < pillar_connection_xy_threshold_) {
                        float z_gap = std::min(std::abs(min_z_j - max_z_i), std::abs(min_z_i - max_z_j));
                        
                        // 如果两个片段在Z方向上有间隙但又不是太远，认为它们是同一柱子的不同部分
                        if (z_gap < pillar_connection_max_z_gap_) {
                            // 合并点云
                            *complete_pillar += *potential_pillars[j];
                            processed[j] = true;
                            found_connection = true;
                            
                            // 更新合并后柱子的高度范围
                            min_z_i = std::min(min_z_i, min_z_j);
                            max_z_i = std::max(max_z_i, max_z_j);
                            
                            // 更新质心（简单平均）
                            x_center = (x_center + x_j) / 2.0f;
                            y_center = (y_center + y_j) / 2.0f;
                            
                            // 填补间隙
                            if (z_gap > z_gap_fill_threshold_) {
                                // 计算柱子的XY平面半径
                                float radius = 0.0f;
                                for (const auto& pt : complete_pillar->points) {
                                    float dx = pt.x - x_center;
                                    float dy = pt.y - y_center;
                                    float r = std::sqrt(dx*dx + dy*dy);
                                    if (r > radius) radius = r;
                                }
                                
                                // 确定Z方向的填充点分布
                                float start_z = std::min(max_z_i, max_z_j);
                                float end_z = std::max(min_z_i, min_z_j);
                                float z_step = accumulation_voxel_size_ * 0.5; // 填充点的Z方向间距
                                
                                // 添加填充点，连接两个片段
                                for (float z = start_z + z_step; z < end_z; z += z_step) {
                                    for (int k = 0; k < pillar_enhancement_points_; ++k) {
                                        float angle = k * (2.0 * M_PI / pillar_enhancement_points_);
                                        float r = radius * 0.8f; // 稍微向内
                                        
                                        PointT new_pt;
                                        new_pt.x = x_center + r * cos(angle);
                                        new_pt.y = y_center + r * sin(angle);
                                        new_pt.z = z;
                                        complete_pillar->points.push_back(new_pt);
                                    }
                                }
                                
                                // 更新点云大小
                                complete_pillar->width = complete_pillar->points.size();
                                complete_pillar->height = 1;
                                complete_pillar->is_dense = false;
                            }
                            
                            // 重新开始搜索，因为合并后的柱子可能与其他片段相连
                            break;
                        }
                    }
                }
            }
            
            // 对完整柱子进行增强
            Eigen::Vector4f centroid;
            centroid[0] = x_center;
            centroid[1] = y_center;
            centroid[2] = (min_z_i + max_z_i) / 2.0f;
            centroid[3] = 0.0f;
            
            PointCloudT::Ptr enhanced_pillar(new PointCloudT);
            enhanceThinPillar(complete_pillar, enhanced_pillar, centroid);
            
            // 将增强后的完整柱子添加到结果中
            *result += *enhanced_pillar;
        }
    }
    
    // 清除积累的点云，重置状态
    void clearAccumulatedClouds() {
        std::lock_guard<std::mutex> lock(cloud_mutex_);
        accumulated_clouds_.clear();
        ROS_INFO("已清除积累点云队列");
    }
    
    // 点云积累相关
    std::deque<PointCloudT::Ptr> accumulated_clouds_;
    std::mutex cloud_mutex_;
    std_msgs::Header latest_header_;
    int max_accumulated_clouds_;
    double accumulation_voxel_size_;
    bool enable_accumulation_;
    double processing_timeout_;  // 处理超时阈值（秒）
    std::atomic<int> processing_count_; // 处理计数器，用于监控活动状态
    
    // 柱子保留和自适应采样相关
    bool enable_pillar_preservation_ = true;   // 是否启用柱子保留优化
    bool enable_adaptive_sampling_ = true;     // 是否启用自适应降采样
    bool enable_connect_broken_pillars_ = true; // 是否启用断开柱子连接
    float pillar_density_factor_ = 0.6;        // 柱子密度因子，值越大，添加的点越多
    int pillar_enhancement_points_ = 4;        // 每个缺失层添加的点数
    float fine_voxel_factor_ = 0.5;            // 精细体素大小因子（相对于标准体素大小）
    float pillar_connection_xy_threshold_ = 0.2; // 连接断开柱子的XY平面距离阈值
    float pillar_connection_max_z_gap_ = 0.8;   // 连接断开柱子的最大Z方向间隙
    float z_gap_fill_threshold_ = 0.1;         // 大于此值的Z方向间隙将被填充点
    
    // 统计输出相关参数
    bool enable_point_count_output_ = true;    // 是否启用点数统计输出
    int detailed_stats_interval_ = 10;         // 详细统计信息输出间隔
    bool enable_detailed_pillar_info_ = true;  // 是否启用详细柱子信息输出
    
    // 混合检测模式相关参数
    bool enable_hybrid_detection_ = true;      // 是否启用混合检测模式
    int min_observation_to_confirm_ = 2;       // 确认为真实柱子需要的最小观测次数
    float pillar_merge_distance_ = 0.3;        // 柱子合并距离阈值
    bool enable_exploration_ = true;           // 是否启用新区域探索
    float known_region_expansion_ = 0.5;       // 已知区域扩展半径
    float pillar_z_search_margin_ = 1.0;       // 柱子Z方向搜索边界扩展
    float new_pillar_height_factor_ = 0.9;     // 新柱子高度因子
    
    // 全局柱子地图相关
    std::map<int, GlobalPillar> global_pillar_map_;  // 全局柱子地图
    pcl::KdTreeFLANN<pcl::PointXYZ> pillar_kdtree_; // 柱子位置KDTree
    PointCloudT::Ptr pillar_centroids_cloud_;        // 柱子质心点云（用于KDTree）
    int next_pillar_id_ = 1;                         // 下一个柱子ID
    mutable std::mutex pillar_map_mutex_;            // 全局地图互斥锁
    
    // 点云预处理参数
    double voxel_size_;
    double height_min_;
    double height_max_;
    
    // 聚类参数
    double cluster_tolerance_;
    int min_cluster_size_;
    int max_cluster_size_;
    
    // 柱子检测参数
    double min_pillar_height_;
    double max_pillar_radius_;
};

int main(int argc, char** argv) {
    // 设置本地化环境为中文UTF-8
    setlocale(LC_ALL, "zh_CN.UTF-8");
    
    ros::init(argc, argv, "pillar_detector");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    ROS_INFO("启动柱子检测节点...");
    PillarDetector detector(nh);
    ROS_INFO("柱子检测节点已准备就绪，等待点云数据...");

    // 添加健康检测定时器，每秒检查一次节点状态
    bool enable_watchdog = true;
    private_nh.param<bool>("enable_watchdog", enable_watchdog, true);
    
    ros::Timer watchdog_timer;
    if (enable_watchdog) {
        watchdog_timer = nh.createTimer(ros::Duration(1.0), [&](const ros::TimerEvent&) {
            // 这个简单的看门狗不做任何特别的事情，但可以用来确保节点仍然响应
            ROS_DEBUG("柱子检测节点运行中...");
        });
    }
    
    // 使用多线程回调队列提高并发性能
    ros::AsyncSpinner spinner(2); // 使用2个线程
    spinner.start();
    ros::waitForShutdown();
    
    ROS_INFO("柱子检测节点正常退出");
    return 0;
}