# Ground_extraction
This code can solve the problem of extracting ground point clouds from complex mountain point cloud data. The code includes CSF, VDVI, and Nearest Neighbor Interpolation to fill holes.

**algorithm procedure:**
1. CSF algorithm to remove high vegetation and buildings.
2. VDVI green leaf index removes low vegetation.
3. The nearest neighbor interpolation fills the hole.

### Usage
#### requirement
we test on the VS2022, PCL 1.14.0, CMake 4.1.1, and 'CSF' .
Please refer to this project to configure 'CSF' dependency:
https://github.com/jianboqi/CSF

#### results


