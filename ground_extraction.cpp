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
* ���ܣ�������ɽ�ص��ε����˲�����
* ���̣�
* 1. CSF�����˲��㷨ȥ����ֲ���뽨����
* 2. VDVI��Ҷָ��ȥ����ֲ��
* 3. ����ڲ�ֵ��ն�(kdtree����������)
* �����
* 1. csf_groundPointCloud.pcd - CSF�˲���ĵ������
* 2. csf_nonGroundPointCloud.pcd - CSF�˲���ķǵ������
* 3. ground_without_vegetation.pcd - VDVIȥ����ֲ����ĵ���
* 4. vegetation_points.pcd - ֲ������
* 5. final_ground_points.pcd - ���ղ�ֵ��������������
* �������ͼ���
* Line 
* 
* 
*/

using namespace std;

using PointT = pcl::PointXYZRGB;

// ��һ����0~255
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

// �����䷽���ȡǿ�ȷָ���ֵ
int CalculateThresholdByOTSU(std::vector<int>& vdvi)
{
    // ǿ��ֱ��ͼ
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

    // ��������
    double sumIntensity = 0.0;
    for (int k = minIntensity; k <= maxIntensity; k++)
    {
        sumIntensity += (double)k * (double)histogramIntensity[k];
    }

    // ��������
    int thrIntensity = 1;
    double otsu = 1;
    int w0 = 0; // ǰ������
    double sumFore = 0.0; // ǰ��������

    for (int k = minIntensity; k <= maxIntensity; k++)
    {
        w0 += histogramIntensity[k];
        int w1 = pcCount - w0; // �󾰵���

        if (w0 == 0) continue;
        if (w1 == 0) break;

        sumFore += (double)k * histogramIntensity[k];
        double u0 = sumFore / w0; // ǰ��ƽ���Ҷ�
        double u1 = (sumIntensity - sumFore) / w1; // ����ƽ���Ҷ�
        double g = (double)w0 * (double)w1 * (u0 - u1) * (u0 - u1); // ��䷽��

        if (g > otsu)
        {
            otsu = g;
            thrIntensity = k;
        }
    }

    return thrIntensity;
}

// ����ڲ�ֵ��ն�
pcl::PointCloud<PointT>::Ptr nearestNeighborInterpolation(
    pcl::PointCloud<PointT>::Ptr inputCloud,
    pcl::PointCloud<PointT>::Ptr groundCloud,
    double radius = 1.0, int maxNeighbors = 5)
{
    pcl::PointCloud<PointT>::Ptr resultCloud(new pcl::PointCloud<PointT>);
    *resultCloud = *groundCloud; // ��ʼ��Ϊ�������

    // ����KD���������������
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(groundCloud);

    // �ҳ�����ɾ�ĵ���㣨����������е����ڵ�ǰ��������еĵ㣩
    pcl::PointCloud<PointT>::Ptr missingPoints(new pcl::PointCloud<PointT>);

    // ����������Ƶ�KD�����ڿ��ٲ���
    pcl::search::KdTree<PointT>::Ptr groundTree(new pcl::search::KdTree<PointT>);
    groundTree->setInputCloud(groundCloud);

    for (size_t i = 0; i < inputCloud->size(); ++i)
    {
        PointT point = inputCloud->points[i];
        std::vector<int> pointIdxNKNSearch(1);
        std::vector<float> pointNKNSquaredDistance(1);

        // ��groundCloud�в��������
        if (groundTree->nearestKSearch(point, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
        {
            // ������������ֵ������Ϊ����㱻��ɾ��
            if (pointNKNSquaredDistance[0] > 0.1) // 10cm��ֵ
            {
                missingPoints->push_back(point);
            }
        }
    }

    cout << "���� " << missingPoints->size() << " �����ܱ���ɾ�ĵ�" << endl;

    // ��ÿ��ȱʧ����в�ֵ
    for (size_t i = 0; i < missingPoints->size(); ++i)
    {
        PointT point = missingPoints->points[i];
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;

        // ��groundCloud�������뾶�ڵ��ھ�
        if (tree->radiusSearch(point, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
        {
            // �����ھӵ��ƽ���߳�
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

                // �����µĵ㣨ʹ��ԭʼ���XY���꣬��ֵ�õ�Z���꣩
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

    cout << "��ֵ��� " << resultCloud->size() - groundCloud->size() << " ����" << endl;

    return resultCloud;
}

int main()
{
    // ==================== ����1: CSF�����˲� ====================
    cout << "=== ����1: CSF�����˲���ʼ ===" << endl;

    // ����ԭʼ����
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    if (pcl::io::loadPCDFile<PointT>("C://Users//Lenovo//Desktop//LiDAR_data//lasdata//cloud9.pcd", *cloud) == -1)
    {
        PCL_ERROR("δ�ܶ�ȡ�ļ�\n");
        return -1;
    }
    cout << "ԭʼ���Ƶ���: " << cloud->size() << endl;

    // ת��ΪCSF֧�ֵ�����
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

    // ����CSF����
    csf.params.bSloopSmooth = false;
    csf.params.cloth_resolution = 0.1;
    csf.params.rigidness = 3;
    csf.params.time_step = 0.65;
    csf.params.class_threshold = 0.05;
    csf.params.interations = 600;

    // ִ��CSF�˲�
    pcl::Indices groundIndexes, offGroundIndexes;
    csf.do_filtering(groundIndexes, offGroundIndexes);
    cout << "CSF�㷨����ʱ��: " << time.getTimeSeconds() << "��" << endl;

    // ��ȡ����ͷǵ������
    pcl::PointCloud<PointT>::Ptr groundCloud(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr offGroundCloud(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud(*cloud, groundIndexes, *groundCloud);
    pcl::copyPointCloud(*cloud, offGroundIndexes, *offGroundCloud);

    cout << "CSF������Ƶ���: " << groundCloud->size() << endl;
    cout << "CSF�ǵ�����Ƶ���: " << offGroundCloud->size() << endl;

    // ����CSF���
    pcl::io::savePCDFileBinary("csf_groundPointCloud.pcd", *groundCloud);
    pcl::io::savePCDFileBinary("csf_nonGroundPointCloud.pcd", *offGroundCloud);

    // ==================== ����2: VDVIȥ����ֲ�� ====================
    cout << "\n=== ����2: VDVIȥ����ֲ����ʼ ===" << endl;

    // ����VDVIָ��
    std::vector<double> dVDVIS(groundCloud->size(), 0);
    for (size_t i = 0; i < groundCloud->points.size(); ++i)
    {
        int R = groundCloud->points[i].r;
        int G = groundCloud->points[i].g;
        int B = groundCloud->points[i].b;
        double VDVI = (2 * G - R - B) * 1.0 / (2 * G + R + B);
        if (VDVI >= 0) dVDVIS[i] = VDVI;
    }

    // ��һ����������ֵ
    auto GV = Normallized255(dVDVIS);
    auto threshold = CalculateThresholdByOTSU(GV);
    cout << "VDVI�Զ�������ֵ: " << threshold << endl;

    // ������ֵ��ȡ����
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

    cout << "ȥ����ֲ���������Ƶ���: " << groundWithoutVegetation->size() << endl;
    cout << "ֲ�����Ƶ���: " << vegetationCloud->size() << endl;

    // ����VDVI���
    pcl::io::savePCDFileBinary("ground_without_vegetation.pcd", *groundWithoutVegetation);
    pcl::io::savePCDFileBinary("vegetation_points.pcd", *vegetationCloud);

    // ==================== ����3: ����ڲ�ֵ��ն� ====================
    cout << "\n=== ����3: ����ڲ�ֵ��ն���ʼ ===" << endl;

    pcl::PointCloud<PointT>::Ptr finalGroundCloud = nearestNeighborInterpolation(
        groundCloud, groundWithoutVegetation, 1.0, 5);

    cout << "���յ�����Ƶ���: " << finalGroundCloud->size() << endl;

    // �������ս��
    pcl::io::savePCDFileBinary("final_ground_points.pcd", *finalGroundCloud);

    // ==================== ������ӻ� ====================
    cout << "\n=== ��ʼ���ӻ� ===" << endl;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("�����˲����̽��"));
    viewer->setWindowName("ɽ�ص��ε����˲�����");

    // �����ĸ��ӿ�
    int v1 = 0, v2 = 1, v3 = 2, v4 = 3;
    viewer->createViewPort(0.0, 0.5, 0.5, 1.0, v1);    // ���ϣ�ԭʼ����
    viewer->createViewPort(0.5, 0.5, 1.0, 1.0, v2);    // ���ϣ�CSF���
    viewer->createViewPort(0.0, 0.0, 0.5, 0.5, v3);    // ���£�VDVI���
    viewer->createViewPort(0.5, 0.0, 1.0, 0.5, v4);    // ���£����ս��

    // �ӿ�1: ԭʼ����
    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_original(cloud);
    viewer->addPointCloud<PointT>(cloud, rgb_original, "original", v1);
    viewer->addText("ԭʼ����", 10, 10, "v1_text", v1);

    // �ӿ�2: CSF����������ɫ���ǵ�����ɫ��
    pcl::visualization::PointCloudColorHandlerCustom<PointT> ground_color(groundCloud, 255, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<PointT> offground_color(offGroundCloud, 0, 255, 0);
    viewer->addPointCloud(groundCloud, ground_color, "csf_ground", v2);
    viewer->addPointCloud(offGroundCloud, offground_color, "csf_offground", v2);
    viewer->addText("CSF�˲����", 10, 10, "v2_text", v2);

    // �ӿ�3: VDVI�������ֲ�����ƣ�
    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_ground(groundWithoutVegetation);
    viewer->addPointCloud(groundWithoutVegetation, rgb_ground, "vdvi_result", v3);
    viewer->addText("VDVIȥ����ֲ��", 10, 10, "v3_text", v3);

    // �ӿ�4: ���ս��
    pcl::visualization::PointCloudColorHandlerCustom<PointT> final_color(finalGroundCloud, 0, 0, 255);
    viewer->addPointCloud(finalGroundCloud, final_color, "final_result", v4);
    viewer->addText("���յ������", 10, 10, "v4_text", v4);

    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    cout << "���ӻ������Ѵ򿪣���q�˳�..." << endl;

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }

    return 0;
}