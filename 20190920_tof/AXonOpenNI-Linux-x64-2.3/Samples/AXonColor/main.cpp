/*****************************************************************************
*                                                                            *
*  OpenNI 2.x Alpha                                                          *
*  Copyright (C) 2012 PrimeSense Ltd.                                        *
*                                                                            *
*  This file is part of OpenNI.                                              *
*                                                                            *
*  Licensed under the Apache License, Version 2.0 (the "License");           *
*  you may not use this file except in compliance with the License.          *
*  You may obtain a copy of the License at                                   *
*                                                                            *
*      http://www.apache.org/licenses/LICENSE-2.0                            *
*                                                                            *
*  Unless required by applicable law or agreed to in writing, software       *
*  distributed under the License is distributed on an "AS IS" BASIS,         *
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
*  See the License for the specific language governing permissions and       *
*  limitations under the License.                                            *
*                                                                            *
*****************************************************************************/
#include <OpenNI.h>
#include "Viewer.h"

int main(int argc, char** argv)
{
	openni::Status rc = openni::STATUS_OK;

	openni::Device device;
	openni::VideoStream depth, color;
	const char* deviceURI = openni::ANY_DEVICE;
	if (argc > 1)
	{
		deviceURI = argv[1];
	}

	rc = openni::OpenNI::initialize();

	printf("After initialization:\n%s\n", openni::OpenNI::getExtendedError());

	rc = device.open(deviceURI);
	if (rc != openni::STATUS_OK)
	{
		printf("SimpleViewer: Device open failed:\n%s\n", openni::OpenNI::getExtendedError());
		openni::OpenNI::shutdown();
		return 1;
	}

	const openni::SensorInfo* info = device.getSensorInfo(openni::SENSOR_COLOR);
	if(info)
	{
		for(int i = 0; i < info->getSupportedVideoModes().getSize(); i++)
			printf("Color info video %d %dx%d FPS %d f %d\n", i,
						info->getSupportedVideoModes()[i].getResolutionX(),
						info->getSupportedVideoModes()[i].getResolutionY(),
						info->getSupportedVideoModes()[i].getFps(),
						info->getSupportedVideoModes()[i].getPixelFormat());
	}
	// rc = depth.create(device, openni::SENSOR_DEPTH);
	// if (rc == openni::STATUS_OK)
	// {
	// 	rc = depth.start();
	// 	if (rc != openni::STATUS_OK)
	// 	{
	// 		printf("SimpleViewer: Couldn't start depth stream:\n%s\n", openni::OpenNI::getExtendedError());
	// 		depth.destroy();
	// 	}
	// }
	// else
	// {
	// 	printf("SimpleViewer: Couldn't find depth stream:\n%s\n", openni::OpenNI::getExtendedError());
	// }

	rc = color.create(device, openni::SENSOR_COLOR);
	if (rc == openni::STATUS_OK)
	{
        openni::VideoMode vm = color.getVideoMode();
        printf("current video mode %d %dX%d fps %d\n",
                                                    vm.getPixelFormat(),
                                                    vm.getResolutionX(),
                                                    vm.getResolutionY(),
                                                    vm.getFps());
        printf("try to set 720p\n");

        vm.setPixelFormat(openni::PIXEL_FORMAT_RGB888);
        vm.setResolution(1280, 720);
        color.setVideoMode(vm);

		rc = color.start();
		if (rc != openni::STATUS_OK)
		{
			printf("SimpleViewer: Couldn't start color stream:\n%s\n", openni::OpenNI::getExtendedError());
			color.destroy();
		}
	}
	else
	{
		printf("SimpleViewer: Couldn't find color stream:\n%s\n", openni::OpenNI::getExtendedError());
	}

	if (!color.isValid())
	{
		printf("SimpleViewer: No valid streams. Exiting\n");
		openni::OpenNI::shutdown();
		return 2;
	}

	SampleViewer sampleViewer("AXon Color Viewer", device, color);

	rc = sampleViewer.init(argc, argv);
	if (rc != openni::STATUS_OK)
	{
		openni::OpenNI::shutdown();
		return 3;
	}
	sampleViewer.run();
}
