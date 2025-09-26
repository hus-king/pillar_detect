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
#include <Eigen/Dense>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

class PillarDetector {
public:
    PillarDetector(ros::NodeHandle& nh) : nh_(nh) {
        // 订阅输入点云
        sub_cloud_ = nh_.subscribe("/cloud_registered", 1, &PillarDetector::cloudCallback, this);
        // 发布检测到的杆子点云
        pub_pillar_ = nh_.advertise<sensor_msgs::PointCloud2>("/cloud_pillar", 1);

        ROS_INFO("柱子检测器已初始化");
        ROS_INFO("订阅话题: /cloud_registered");
        ROS_INFO("发布话题: /cloud_pillar");
    }

private:
    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        // 转换为 PCL 点云
        PointCloudT::Ptr cloud(new PointCloudT);
        pcl::fromROSMsg(*msg, *cloud);

        if (cloud->empty()) {
            ROS_WARN("接收到空点云，跳过处理");
            return;
        }

        ROS_DEBUG("接收到点云，包含 %zu 个点", cloud->size());

        // 1. 体素滤波降采样（可选，加速处理）
        PointCloudT::Ptr cloud_filtered(new PointCloudT);
        pcl::VoxelGrid<PointT> voxel;
        voxel.setInputCloud(cloud);
        voxel.setLeafSize(0.05f, 0.05f, 0.05f); // 5cm 体素
        voxel.filter(*cloud_filtered);

        // 2. 高度滤波：只保留一定高度范围内的点（例如 0.5m ~ 3.0m）
        PointCloudT::Ptr cloud_height(new PointCloudT);
        pcl::PassThrough<PointT> pass;
        pass.setInputCloud(cloud_filtered);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(0.5, 3.0); // 假设杆子高度在此范围
        pass.filter(*cloud_height);

        if (cloud_height->empty()) {
            ROS_WARN("高度滤波后点云为空，跳过处理");
            return;
        }

        ROS_DEBUG("高度滤波后剩余 %zu 个点", cloud_height->size());

        // 3. 欧氏聚类（基于距离的聚类）
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
        tree->setInputCloud(cloud_height);

        pcl::EuclideanClusterExtraction<PointT> ec;
        ec.setClusterTolerance(0.2); // 20cm 内的点视为同一簇
        ec.setMinClusterSize(20);    // 最小点数（根据密度调整）
        ec.setMaxClusterSize(10000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud_height);
        ec.extract(cluster_indices);

        ROS_INFO("检测到 %zu 个聚类簇", cluster_indices.size());

        // 4. 筛选"杆子"候选簇
        PointCloudT::Ptr pillar_cloud(new PointCloudT);
        int pillar_count = 0;
        for (const auto& indices : cluster_indices) {
            PointCloudT::Ptr cluster(new PointCloudT);
            pcl::copyPointCloud(*cloud_height, indices, *cluster);
            
            // 计算包围盒或主方向
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
            
            // 计算XY平面投影的半径（近似）
            float max_radius = 0.0f;
            for (const auto& pt : cluster->points) {
                float dx = pt.x - centroid[0];
                float dy = pt.y - centroid[1];
                float r = std::sqrt(dx*dx + dy*dy);
                if (r > max_radius) max_radius = r;
            }
            
            // 启发式判断：高而细（height > 1.0m 且 半径 < 0.2m）
            if (height > 1.0 && max_radius < 0.2) {
                // 可选：检查是否"孤立"——周围一定范围内无其他大簇
                // 此处简化：仅用几何特征
                pillar_count++;
                ROS_INFO("检测到柱子 %d: 高度=%.2fm, 半径=%.2fm, 包含%zu个点", 
                         pillar_count, height, max_radius, cluster->size());
                *pillar_cloud += *cluster;
            }
        }

        // 5. 发布结果
        ROS_INFO("本次检测完成，共找到 %d 个柱子，发布 %zu 个点", pillar_count, pillar_cloud->size());
        
        sensor_msgs::PointCloud2 output_msg;
        pcl::toROSMsg(*pillar_cloud, output_msg);
        output_msg.header = msg->header; // 保持时间戳和坐标系
        pub_pillar_.publish(output_msg);
    }

    ros::NodeHandle nh_;
    ros::Subscriber sub_cloud_;
    ros::Publisher pub_pillar_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "pillar_detector");
    ros::NodeHandle nh;

    ROS_INFO("启动柱子检测节点...");
    PillarDetector detector(nh);
    ROS_INFO("柱子检测节点已准备就绪，等待点云数据...");

    ros::spin();
    ROS_INFO("柱子检测节点正常退出");
    return 0;
}