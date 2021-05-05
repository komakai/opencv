// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include <opencv2/rgbd/volume.hpp>

#include "hash_tsdf.hpp"
#include "opencv2/core/base.hpp"
#include "precomp.hpp"
#include "tsdf.hpp"
#include "hash_tsdf.hpp"
#include "colored_tsdf.hpp"

namespace cv
{

Ptr<VolumeParams> VolumeParams::defaultParams(VolumeType _volumeType)
{
    VolumeParams params;
    params.type              = _volumeType;
    params.maxWeight         = 64;
    params.raycastStepFactor = 0.25f;
    params.unitResolution    = 0;  // unitResolution not used for TSDF
    float volumeSize         = 3.0f;
    Matx44f pose = Affine3f().translate(Vec3f(-volumeSize / 2.f, -volumeSize / 2.f, 0.5f)).matrix;
    params.pose00 = pose(0, 0); params.pose01 = pose(0, 1); params.pose02 = pose(0, 2);
    params.pose10 = pose(1, 0); params.pose11 = pose(1, 1); params.pose12 = pose(1, 2);
    params.pose20 = pose(2, 0); params.pose21 = pose(2, 1); params.pose22 = pose(2, 2);
    params.pose03 = pose(0, 3); params.pose13 = pose(1, 3); params.pose23 = pose(2, 3);

    if(params.type == VolumeType::TSDF)
    {
        params.resolutionX = 512;
        params.resolutionY = 512;
        params.resolutionZ = 512;
        params.voxelSize           = volumeSize / 512.f;
        params.depthTruncThreshold = 0.f;  // depthTruncThreshold not required for TSDF
        params.tsdfTruncDist = 7 * params.voxelSize;  //! About 0.04f in meters
        return makePtr<VolumeParams>(params);
    }
    else if(params.type == VolumeType::HASHTSDF)
    {
        params.unitResolution      = 16;
        params.voxelSize           = volumeSize / 512.f;
        params.depthTruncThreshold = Odometry::DEFAULT_MAX_DEPTH();
        params.tsdfTruncDist = 7 * params.voxelSize;  //! About 0.04f in meters
        return makePtr<VolumeParams>(params);
    }
    else if (params.type == VolumeType::COLOREDTSDF)
    {
        params.resolutionX = 512;
        params.resolutionY = 512;
        params.resolutionZ = 512;
        params.voxelSize = volumeSize / 512.f;
        params.depthTruncThreshold = 0.f;  // depthTruncThreshold not required for TSDF
        params.tsdfTruncDist = 7 * params.voxelSize;  //! About 0.04f in meters
        return makePtr<VolumeParams>(params);
    }
    CV_Error(Error::StsBadArg, "Invalid VolumeType does not have parameters");
}

Ptr<VolumeParams> VolumeParams::coarseParams(VolumeType _volumeType)
{
    Ptr<VolumeParams> params = defaultParams(_volumeType);

    params->raycastStepFactor = 0.75f;
    float volumeSize          = 3.0f;
    if(params->type == VolumeType::TSDF)
    {
        params->resolutionX = 128;
        params->resolutionY = 128;
        params->resolutionZ = 128;
        params->voxelSize  = volumeSize / 128.f;
        params->tsdfTruncDist = 2 * params->voxelSize;  //! About 0.04f in meters
        return params;
    }
    else if(params->type == VolumeType::HASHTSDF)
    {
        params->voxelSize = volumeSize / 128.f;
        params->tsdfTruncDist = 2 * params->voxelSize;  //! About 0.04f in meters
        return params;
    }
    else if (params->type == VolumeType::COLOREDTSDF)
    {
        params->resolutionX = 128;
        params->resolutionY = 128;
        params->resolutionZ = 128;
        params->voxelSize = volumeSize / 128.f;
        params->tsdfTruncDist = 2 * params->voxelSize;  //! About 0.04f in meters
        return params;
    }
    CV_Error(Error::StsBadArg, "Invalid VolumeType does not have parameters");
}

Ptr<Volume> makeVolume(const VolumeParams& _volumeParams)
{
    if(_volumeParams.type == VolumeType::TSDF)
        return makeTSDFVolume(_volumeParams);
    else if(_volumeParams.type == VolumeType::HASHTSDF)
        return makeHashTSDFVolume(_volumeParams);
    else if(_volumeParams.type == VolumeType::COLOREDTSDF)
        return makeColoredTSDFVolume(_volumeParams);
    CV_Error(Error::StsBadArg, "Invalid VolumeType does not have parameters");
}

Ptr<Volume> makeVolume(VolumeType _volumeType, float _voxelSize, Matx44f _pose,
                       float _raycastStepFactor, float _truncDist, int _maxWeight,
                       float _truncateThreshold, Point3i _resolution)
{
    Point3i _presolution = _resolution;
    if (_volumeType == VolumeType::TSDF)
    {
        return makeTSDFVolume(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight, _presolution);
    }
    else if (_volumeType == VolumeType::HASHTSDF)
    {
        return makeHashTSDFVolume(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight, _truncateThreshold);
    }
    else if (_volumeType == VolumeType::COLOREDTSDF)
    {
        return makeColoredTSDFVolume(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight, _presolution);
    }
    CV_Error(Error::StsBadArg, "Invalid VolumeType does not have parameters");
}

}  // namespace cv
