#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <CSF.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/time.h>
#include <pcl/octree/octree_search.h>
#include <boost/thread/thread.hpp>
#include <pcl/visualization/pcl_visualizer.h>


/*
* Data: 2025.9.24
* Authors: yekai
* 
* 介绍：复杂山地地形地面点提取方法参考
* 流程：
* 1. CSF布料滤波算法去除高植被与建筑物
* 2. VDVI绿叶指数去除低植被
* 3. 最近邻插值填补空洞
* 输出：
* 1. csf_groundPointCloud.pcd - CSF滤波后的地面点云
* 2. csf_nonGroundPointCloud.pcd - CSF滤波后的非地面点云
* 3. ground_without_vegetation.pcd - VDVI去除低植被后的点云
* 4. vegetation_points.pcd - 植被点云
* 5. final_ground_points.pcd - 最终插值后的完整地面点云
* 
* 参数解释见：
* Line 131-132
* Line 191
* Line 260-265
* 
* 例子：原始点云数为16166028，效率为
*     CSF：8-10 minutes
*     VDVI：<3 s
*     最近邻插值：14-18 minutes
*
*/
using namespace std;

using PointT = pcl::PointXYZRGB;

// 归一化到0~255
std::vector<int> Normallized255(std::vector<double>& gv)
{
    if (gv.empty()) return std::vector<int>();

    double Gmax = *max_element(gv.begin(), gv.end());
    double Gmin = *min_element(gv.begin(), gv.end());

    if (Gmax == Gmin)
    {
        // 所有值相同的情况
        std::vector<int> GV(gv.size(), 128);
        return GV;
    }

    std::vector<int> GV(gv.size(), 0);
    for (size_t i = 0; i < gv.size(); ++i)
    {
        GV[i] = static_cast<int>(255 * (gv[i] - Gmin) / (Gmax - Gmin));
    }
    return GV;
}

// 最大类间方差法获取强度分割阈值
int CalculateThresholdByOTSU(std::vector<int>& vdvi)
{
    if (vdvi.empty()) return 128;

    // 强度直方图
    std::vector<int> histogramIntensity(256, 0); // 使用vector避免栈溢出
    int maxIntensity = INT_MIN, minIntensity = INT_MAX;
    int pcCount = static_cast<int>(vdvi.size());

    for (int i = 0; i < pcCount; ++i)
    {
        int vIntensity = vdvi[i];
        if (vIntensity > 255) vIntensity = 255;
        if (vIntensity < 0) vIntensity = 0;

        if (vIntensity > maxIntensity) maxIntensity = vIntensity;
        if (vIntensity < minIntensity) minIntensity = vIntensity;
        ++histogramIntensity[vIntensity];
    }

    // 总质量矩
    double sumIntensity = 0.0;
    for (int k = minIntensity; k <= maxIntensity; k++)
    {
        sumIntensity += static_cast<double>(k) * static_cast<double>(histogramIntensity[k]);
    }

    // 遍历计算
    int thrIntensity = minIntensity;
    double maxOtsu = -1.0;
    int w0 = 0; // 前景点数
    double sumFore = 0.0; // 前景质量矩

    for (int k = minIntensity; k <= maxIntensity; k++)
    {
        w0 += histogramIntensity[k];
        int w1 = pcCount - w0; // 后景点数

        if (w0 == 0) continue;
        if (w1 == 0) break;

        sumFore += static_cast<double>(k) * histogramIntensity[k];
        double u0 = sumFore / w0; // 前景平均灰度
        double u1 = (sumIntensity - sumFore) / w1; // 背景平均灰度
        double g = static_cast<double>(w0) * static_cast<double>(w1) * (u0 - u1) * (u0 - u1); // 类间方差

        if (g > maxOtsu)
        {
            maxOtsu = g;
            thrIntensity = k;
        }
    }

    return thrIntensity;
}

// 使用八叉树进行最近邻插值填补空洞
pcl::PointCloud<PointT>::Ptr octreeNearestNeighborInterpolation(
    pcl::PointCloud<PointT>::Ptr inputCloud,
    pcl::PointCloud<PointT>::Ptr groundCloud,
    double resolution = 1.0, // 八叉树分辨率
    int maxNeighbors = 5) // 最大邻域点数
{
    pcl::PointCloud<PointT>::Ptr resultCloud(new pcl::PointCloud<PointT>);
    *resultCloud = *groundCloud; // 初始化为地面点云

    if (groundCloud->empty())
    {
        cout << "警告：地面点云为空，无法进行插值" << endl;
        return resultCloud;
    }

    // 创建八叉树
    pcl::octree::OctreePointCloudSearch<PointT> octree(resolution);
    octree.setInputCloud(groundCloud);
    octree.addPointsFromInputCloud();

    cout << "八叉树构建完成，分辨率: " << resolution << endl;

    // 找出被误删的地面点
    pcl::PointCloud<PointT>::Ptr missingPoints(new pcl::PointCloud<PointT>);
    int missingCount = 0;

    for (size_t i = 0; i < inputCloud->size(); ++i)
    {
        PointT point = inputCloud->points[i];

        // 在八叉树中搜索最近点
        std::vector<int> pointIdxVec;
        std::vector<float> pointSquaredDistance;

        if (octree.nearestKSearch(point, 1, pointIdxVec, pointSquaredDistance) > 0)
        {
            // 如果距离大于阈值，则认为这个点被误删了
            if (pointSquaredDistance[0] > 0.1) // 10cm阈值
            {
                missingPoints->push_back(point);
                missingCount++;
            }
        }
    }

    cout << "发现 " << missingCount << " 个可能被误删的点" << endl;

    if (missingPoints->empty())
    {
        cout << "没有发现需要插值的点" << endl;
        return resultCloud;
    }

    // 对每个缺失点进行插值
    int interpolatedCount = 0;

    for (size_t i = 0; i < missingPoints->size(); ++i)
    {
        PointT point = missingPoints->points[i];
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;

        // 在八叉树中搜索半径内的邻居
        double searchRadius = resolution * 2.0; // 搜索半径为分辨率的2倍

        if (octree.radiusSearch(point, searchRadius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
        {
            // 计算邻居点的平均高程
            double avgZ = 0.0;
            int validNeighbors = 0;

            // 限制最大邻居数
            size_t neighborCount = min(pointIdxRadiusSearch.size(), static_cast<size_t>(maxNeighbors));

            for (size_t j = 0; j < neighborCount; ++j)
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
                newPoint.z = static_cast<float>(avgZ);
                newPoint.r = point.r;
                newPoint.g = point.g;
                newPoint.b = point.b;

                resultCloud->push_back(newPoint);
                interpolatedCount++;
            }
        }
    }

    cout << "八叉树插值填补了 " << interpolatedCount << " 个点" << endl;

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

    // 设置CSF参数（根据山地地形调整）
    csf.params.bSloopSmooth = false;     // 是否开启平滑，山地有坡度建议关闭坡度平滑
    csf.params.cloth_resolution = 0.1;   // 布料网格分辨率，值越小网格网格越密集，计算精度高耗时长
    csf.params.rigidness = 2;            // 布料刚度，值越大布料越硬
    csf.params.time_step = 0.65;         // 时间步长，影响模拟的稳定性和收敛速度，值越小越稳定
    csf.params.class_threshold = 0.08;    // 分类阈值，用于区分地面点和非地面点的距离阈值，值越大分类越严格
    csf.params.interations = 600;        // 模拟的最大迭代次数，模拟越稳定耗时越长

    // 执行CSF滤波
    pcl::Indices groundIndexes, offGroundIndexes;
    csf.do_filtering(groundIndexes, offGroundIndexes);
    double csfTime = time.getTimeSeconds();
    cout << "CSF算法运行时间: " << csfTime << "秒" << endl;

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

    if (groundCloud->empty())
    {
        cout << "警告：CSF地面点云为空，跳过VDVI处理" << endl;
        return -1;
    }

    // 计算VDVI指数
    std::vector<double> dVDVIS(groundCloud->size(), 0);
    int validVDVICount = 0;

    for (size_t i = 0; i < groundCloud->points.size(); ++i)
    {
        int R = groundCloud->points[i].r;
        int G = groundCloud->points[i].g;
        int B = groundCloud->points[i].b;

        // 避免除零错误
        if ((2 * G + R + B) == 0)
        {
            dVDVIS[i] = 0;
            continue;
        }

        double VDVI = (2.0 * G - R - B) / (2.0 * G + R + B);
        dVDVIS[i] = VDVI;
        validVDVICount++;
    }

    cout << "有效VDVI计算点数: " << validVDVICount << endl;

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

    // ==================== 步骤3: 八叉树最近邻插值填补空洞 ====================
    cout << "\n=== 步骤3: 八叉树最近邻插值填补空洞开始 ===" << endl;

    pcl::PointCloud<PointT>::Ptr finalGroundCloud = octreeNearestNeighborInterpolation(
        groundCloud, groundWithoutVegetation, 0.5, 5);

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

    // 设置背景色
    viewer->setBackgroundColor(0.05, 0.05, 0.05, v1);
    viewer->setBackgroundColor(0.05, 0.05, 0.05, v2);
    viewer->setBackgroundColor(0.05, 0.05, 0.05, v3);
    viewer->setBackgroundColor(0.05, 0.05, 0.05, v4);

    cout << "可视化窗口已打开，按q退出..." << endl;

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }

    cout << "程序执行完成！" << endl;
    cout << "输出文件：" << endl;
    cout << "1. csf_groundPointCloud.pcd - CSF滤波地面点" << endl;
    cout << "2. csf_nonGroundPointCloud.pcd - CSF滤波非地面点" << endl;
    cout << "3. ground_without_vegetation.pcd - VDVI去除植被后地面点" << endl;
    cout << "4. vegetation_points.pcd - 植被点" << endl;
    cout << "5. final_ground_points.pcd - 最终插值后地面点" << endl;

    return 0;
}

