#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <CSF.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/time.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <boost/thread/thread.hpp>
#include <pcl/visualization/pcl_visualizer.h>

/*
* Data: 2025.9.24
* Authors: yekai
* 功能：完整的山地地形点云滤波流程
* 流程：
* 1. CSF布料滤波算法去除高植被与建筑物
* 2. VDVI绿叶指数去除低植被
* 3. 最近邻插值填补空洞(kdtree检索近邻域)
* 输出：
* 1. csf_groundPointCloud.pcd - CSF滤波后的地面点云
* 2. csf_nonGroundPointCloud.pcd - CSF滤波后的非地面点云
* 3. ground_without_vegetation.pcd - VDVI去除低植被后的点云
* 4. vegetation_points.pcd - 植被点云
* 5. final_ground_points.pcd - 最终插值后的完整地面点云
* 参数解释见：
* Line 
* 
* 
*/

using namespace std;

using PointT = pcl::PointXYZRGB;

// 归一化到0~255
std::vector<int> Normallized255(std::vector<double>& gv)
{
    double Gmax = *max_element(gv.begin(), gv.end());
    double Gmin = *min_element(gv.begin(), gv.end());
    std::vector<int> GV(gv.size(), 0);
    for (size_t i = 0; i < gv.size(); ++i)
    {
        GV[i] = 255 * (gv[i] - Gmin) / (Gmax - Gmin);
    }
    return GV;
}

// 最大类间方差法获取强度分割阈值
int CalculateThresholdByOTSU(std::vector<int>& vdvi)
{
    // 强度直方图
    int histogramIntensity[65536] = { 0 };
    int maxIntensity = INT64_MIN, minIntensity = INT64_MAX;
    int pcCount = vdvi.size();

    for (int i = 0; i < pcCount; ++i)
    {
        int vIntensity = vdvi[i];
        if (vIntensity > maxIntensity) maxIntensity = vIntensity;
        if (vIntensity < minIntensity) minIntensity = vIntensity;
        ++histogramIntensity[vIntensity];
    }

    // 总质量矩
    double sumIntensity = 0.0;
    for (int k = minIntensity; k <= maxIntensity; k++)
    {
        sumIntensity += (double)k * (double)histogramIntensity[k];
    }

    // 遍历计算
    int thrIntensity = 1;
    double otsu = 1;
    int w0 = 0; // 前景点数
    double sumFore = 0.0; // 前景质量矩

    for (int k = minIntensity; k <= maxIntensity; k++)
    {
        w0 += histogramIntensity[k];
        int w1 = pcCount - w0; // 后景点数

        if (w0 == 0) continue;
        if (w1 == 0) break;

        sumFore += (double)k * histogramIntensity[k];
        double u0 = sumFore / w0; // 前景平均灰度
        double u1 = (sumIntensity - sumFore) / w1; // 背景平均灰度
        double g = (double)w0 * (double)w1 * (u0 - u1) * (u0 - u1); // 类间方差

        if (g > otsu)
        {
            otsu = g;
            thrIntensity = k;
        }
    }

    return thrIntensity;
}

// 最近邻插值填补空洞
pcl::PointCloud<PointT>::Ptr nearestNeighborInterpolation(
    pcl::PointCloud<PointT>::Ptr inputCloud,
    pcl::PointCloud<PointT>::Ptr groundCloud,
    double radius = 1.0, int maxNeighbors = 5)
{
    pcl::PointCloud<PointT>::Ptr resultCloud(new pcl::PointCloud<PointT>);
    *resultCloud = *groundCloud; // 初始化为地面点云

    // 创建KD树用于最近邻搜索
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(groundCloud);

    // 找出被误删的地面点（在输入点云中但不在当前地面点云中的点）
    pcl::PointCloud<PointT>::Ptr missingPoints(new pcl::PointCloud<PointT>);

    // 创建地面点云的KD树用于快速查找
    pcl::search::KdTree<PointT>::Ptr groundTree(new pcl::search::KdTree<PointT>);
    groundTree->setInputCloud(groundCloud);

    for (size_t i = 0; i < inputCloud->size(); ++i)
    {
        PointT point = inputCloud->points[i];
        std::vector<int> pointIdxNKNSearch(1);
        std::vector<float> pointNKNSquaredDistance(1);

        // 在groundCloud中查找最近点
        if (groundTree->nearestKSearch(point, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
        {
            // 如果距离大于阈值，则认为这个点被误删了
            if (pointNKNSquaredDistance[0] > 0.1) // 10cm阈值
            {
                missingPoints->push_back(point);
            }
        }
    }

    cout << "发现 " << missingPoints->size() << " 个可能被误删的点" << endl;

    // 对每个缺失点进行插值
    for (size_t i = 0; i < missingPoints->size(); ++i)
    {
        PointT point = missingPoints->points[i];
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;

        // 在groundCloud中搜索半径内的邻居
        if (tree->radiusSearch(point, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
        {
            // 计算邻居点的平均高程
            double avgZ = 0.0;
            int validNeighbors = 0;

            for (size_t j = 0; j < pointIdxRadiusSearch.size() && j < (size_t)maxNeighbors; ++j)
            {
                avgZ += groundCloud->points[pointIdxRadiusSearch[j]].z;
                validNeighbors++;
            }

            if (validNeighbors > 0)
            {
                avgZ /= validNeighbors;

                // 创建新的点（使用原始点的XY坐标，插值得到Z坐标）
                PointT newPoint;
                newPoint.x = point.x;
                newPoint.y = point.y;
                newPoint.z = avgZ;
                newPoint.r = point.r;
                newPoint.g = point.g;
                newPoint.b = point.b;

                resultCloud->push_back(newPoint);
            }
        }
    }

    cout << "插值填补了 " << resultCloud->size() - groundCloud->size() << " 个点" << endl;

    return resultCloud;
}

int main()
{
    // ==================== 步骤1: CSF布料滤波 ====================
    cout << "=== 步骤1: CSF布料滤波开始 ===" << endl;

    // 加载原始点云
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    if (pcl::io::loadPCDFile<PointT>("C://Users//Lenovo//Desktop//LiDAR_data//lasdata//cloud9.pcd", *cloud) == -1)
    {
        PCL_ERROR("未能读取文件\n");
        return -1;
    }
    cout << "原始点云点数: " << cloud->size() << endl;

    // 转换为CSF支持的数据
    vector<csf::Point> CSFcloud(cloud->size());
    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        CSFcloud[i].x = cloud->points[i].x;
        CSFcloud[i].y = cloud->points[i].y;
        CSFcloud[i].z = cloud->points[i].z;
    }

    pcl::StopWatch time;
    CSF csf;
    csf.setPointCloud(CSFcloud);

    // 设置CSF参数
    csf.params.bSloopSmooth = false;
    csf.params.cloth_resolution = 0.1;
    csf.params.rigidness = 3;
    csf.params.time_step = 0.65;
    csf.params.class_threshold = 0.05;
    csf.params.interations = 600;

    // 执行CSF滤波
    pcl::Indices groundIndexes, offGroundIndexes;
    csf.do_filtering(groundIndexes, offGroundIndexes);
    cout << "CSF算法运行时间: " << time.getTimeSeconds() << "秒" << endl;

    // 提取地面和非地面点云
    pcl::PointCloud<PointT>::Ptr groundCloud(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr offGroundCloud(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud(*cloud, groundIndexes, *groundCloud);
    pcl::copyPointCloud(*cloud, offGroundIndexes, *offGroundCloud);

    cout << "CSF地面点云点数: " << groundCloud->size() << endl;
    cout << "CSF非地面点云点数: " << offGroundCloud->size() << endl;

    // 保存CSF结果
    pcl::io::savePCDFileBinary("csf_groundPointCloud.pcd", *groundCloud);
    pcl::io::savePCDFileBinary("csf_nonGroundPointCloud.pcd", *offGroundCloud);

    // ==================== 步骤2: VDVI去除低植被 ====================
    cout << "\n=== 步骤2: VDVI去除低植被开始 ===" << endl;

    // 计算VDVI指数
    std::vector<double> dVDVIS(groundCloud->size(), 0);
    for (size_t i = 0; i < groundCloud->points.size(); ++i)
    {
        int R = groundCloud->points[i].r;
        int G = groundCloud->points[i].g;
        int B = groundCloud->points[i].b;
        double VDVI = (2 * G - R - B) * 1.0 / (2 * G + R + B);
        if (VDVI >= 0) dVDVIS[i] = VDVI;
    }

    // 归一化并计算阈值
    auto GV = Normallized255(dVDVIS);
    auto threshold = CalculateThresholdByOTSU(GV);
    cout << "VDVI自动计算阈值: " << threshold << endl;

    // 根据阈值提取点云
    pcl::PointCloud<PointT>::Ptr vegetationCloud(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr groundWithoutVegetation(new pcl::PointCloud<PointT>);

    for (size_t i = 0; i < GV.size(); ++i)
    {
        if (GV[i] > threshold)
        {
            vegetationCloud->push_back(groundCloud->points[i]);
        }
        else
        {
            groundWithoutVegetation->push_back(groundCloud->points[i]);
        }
    }

    cout << "去除低植被后地面点云点数: " << groundWithoutVegetation->size() << endl;
    cout << "植被点云点数: " << vegetationCloud->size() << endl;

    // 保存VDVI结果
    pcl::io::savePCDFileBinary("ground_without_vegetation.pcd", *groundWithoutVegetation);
    pcl::io::savePCDFileBinary("vegetation_points.pcd", *vegetationCloud);

    // ==================== 步骤3: 最近邻插值填补空洞 ====================
    cout << "\n=== 步骤3: 最近邻插值填补空洞开始 ===" << endl;

    pcl::PointCloud<PointT>::Ptr finalGroundCloud = nearestNeighborInterpolation(
        groundCloud, groundWithoutVegetation, 1.0, 5);

    cout << "最终地面点云点数: " << finalGroundCloud->size() << endl;

    // 保存最终结果
    pcl::io::savePCDFileBinary("final_ground_points.pcd", *finalGroundCloud);

    // ==================== 结果可视化 ====================
    cout << "\n=== 开始可视化 ===" << endl;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("完整滤波流程结果"));
    viewer->setWindowName("山地地形点云滤波流程");

    // 创建四个视口
    int v1 = 0, v2 = 1, v3 = 2, v4 = 3;
    viewer->createViewPort(0.0, 0.5, 0.5, 1.0, v1);    // 左上：原始点云
    viewer->createViewPort(0.5, 0.5, 1.0, 1.0, v2);    // 右上：CSF结果
    viewer->createViewPort(0.0, 0.0, 0.5, 0.5, v3);    // 左下：VDVI结果
    viewer->createViewPort(0.5, 0.0, 1.0, 0.5, v4);    // 右下：最终结果

    // 视口1: 原始点云
    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_original(cloud);
    viewer->addPointCloud<PointT>(cloud, rgb_original, "original", v1);
    viewer->addText("原始点云", 10, 10, "v1_text", v1);

    // 视口2: CSF结果（地面红色，非地面绿色）
    pcl::visualization::PointCloudColorHandlerCustom<PointT> ground_color(groundCloud, 255, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<PointT> offground_color(offGroundCloud, 0, 255, 0);
    viewer->addPointCloud(groundCloud, ground_color, "csf_ground", v2);
    viewer->addPointCloud(offGroundCloud, offground_color, "csf_offground", v2);
    viewer->addText("CSF滤波结果", 10, 10, "v2_text", v2);

    // 视口3: VDVI结果（非植被点云）
    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_ground(groundWithoutVegetation);
    viewer->addPointCloud(groundWithoutVegetation, rgb_ground, "vdvi_result", v3);
    viewer->addText("VDVI去除低植被", 10, 10, "v3_text", v3);

    // 视口4: 最终结果
    pcl::visualization::PointCloudColorHandlerCustom<PointT> final_color(finalGroundCloud, 0, 0, 255);
    viewer->addPointCloud(finalGroundCloud, final_color, "final_result", v4);
    viewer->addText("最终地面点云", 10, 10, "v4_text", v4);

    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    cout << "可视化窗口已打开，按q退出..." << endl;

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }

    return 0;
}