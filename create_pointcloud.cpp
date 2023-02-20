bool PTRealsense2::extract3dPoints()
{
    float depthPixel[2];
    float distance;
    float depthPoint[3];

    for (int r = 0; r < _depthMat.rows; r++) {
        for (int c = 0; c < _depthMat.cols; c++) {
            depthPixel[0] = (float)c;
			depthPixel[1] = (float)r;

            distance = (float)_depthMat.at<uint16_t>(r, c) * _depth_scale;
            rs2_deproject_pixel_to_point(depthPoint, &_depthIntrinsics, depthPixel, distance);
           // std::cout<<r<<std::endl;



            _pointCloud.x[r * _depthMat.cols + c] = -(signed short)(depthPoint[0] * 1000);
            _pointCloud.y[r * _depthMat.cols + c] = -(signed short)(depthPoint[1] * 1000);
            _pointCloud.z[r * _depthMat.cols + c] = (signed short)(depthPoint[2] * 1000);




        }
    }





    _pointCloud.size = _depthMat.rows * _depthMat.cols;
    _pointCloudCalculated = true;
    return true;
}
