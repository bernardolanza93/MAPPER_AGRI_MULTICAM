from open_gopro import WiredGoPro, Params , proto, WirelessGoPro

import asyncio
from misc_utility_library import *
from open_gopro.constants import WebcamError, WebcamStatus
from open_gopro.demos.gui.components.util import display_video_blocking
from open_gopro.gopro_base import GoProBase
from open_gopro.logger import setup_logging
from open_gopro.util import add_cli_args_and_parse
from typing import Final

TIME_ACQUISITION = 0
PHOTO_GOPRO = 0

WEBCAM = 1
now_GP = datetime.now()
timing_GP = now_GP.strftime("%Y_%m_%d_%H_%M_%S")
video_string = "aquisition/" +"GP_"+ timing_GP + "_vid.mp4"
photo_string = "aquisition/" +"GP_"+ timing_GP + "_img.jpg"
DOWNLOAD_ALL = 0


STREAM_URL: Final[str] = r"udp://0.0.0.0:8554"




async def wait_for_webcam_status(gopro: GoProBase, statuses: set[WebcamStatus], timeout: int = 10) -> bool:
    """Wait for specified webcam status(es) for a given timeout

    Args:
        gopro (GoProBase): gopro to communicate with
        statuses (set[WebcamStatus]): statuses to wait for
        timeout (int): timeout in seconds. Defaults to 10.

    Returns:
        bool: True if status was received before timing out, False if timed out or received error
    """

    async def poll_for_status() -> bool:
        # Poll until status is received
        while True:
            response = (await gopro.http_command.webcam_status()).data
            if response.error != WebcamError.SUCCESS:
                # Something bad happened
                return False
            if response.status in statuses:
                # We found the desired status
                return True

    # Wait for either status or timeout
    try:
        return await asyncio.wait_for(poll_for_status(), timeout)
    except TimeoutError:
        return False

async def main() -> None:
    async with WiredGoPro() as gopro:
        print("Yay! # Put our code hereI'm connected via USB, opened, and ready to send / get data now!")
        # Send some messages now
        print("connected?",gopro.is_http_connected)
        #gopro.http_command.set_shutter(shutter=Params.Toggle.ENABLE)


        info =  gopro.http_command.get_camera_info
        # print(gopro.http_command.get_camera_state)
        # print(gopro.http_command.get_media_list)


        if DOWNLOAD_ALL:
            # Download all of the files from the camera
            media_list = (await gopro.http_command.get_media_list()).data.files
            for item in media_list:
                await gopro.http_command.download_file(camera_file=item.filename)

        if PHOTO_GOPRO:
            assert gopro
            # Configure settings to prepare for photo
            await gopro.http_setting.video_performance_mode.set(Params.PerformanceMode.MAX_PERFORMANCE)
            await gopro.http_setting.max_lens_mode.set(Params.MaxLensMode.DEFAULT)
            await gopro.http_setting.camera_ux_mode.set(Params.CameraUxMode.PRO)
            await gopro.http_setting.resolution.set(Params.Resolution.RES_1080)
            await gopro.http_command.set_turbo_mode(mode=Params.Toggle.DISABLE)
            assert (await gopro.http_command.load_preset_group(group=proto.EnumPresetGroup.PRESET_GROUP_ID_PHOTO)).ok

            media_set_before = set((await gopro.http_command.get_media_list()).data.files)
            # Take a photo
            print("Capturing a photo...")
            assert (await gopro.http_command.set_shutter(shutter=Params.Toggle.ENABLE)).ok

            time.sleep(5)

            # Get the media list after
            media_set_after = set((await gopro.http_command.get_media_list()).data.files)
            # The photo is (most likely) the difference between the two sets
            photo = media_set_after.difference(media_set_before).pop()
            # Download the photo
            print(f"Downloading {photo.filename}...")
            await gopro.http_command.download_file(camera_file=photo.filename, local_file=photo_string)
            print(f"Success!! :smiley: File has been downloaded to {photo_string}")

            if gopro:
                await gopro.close()
            print("Exiting...")


        if TIME_ACQUISITION:







            assert gopro
            # Configure settings to prepare for video
            #REGOLA FPS E RISOLUZIONE
            await gopro.http_setting.fps.set(Params.FPS.FPS_120)
            await gopro.http_setting.resolution.set(Params.Resolution.RES_1080)

            await gopro.http_setting.video_performance_mode.set(Params.PerformanceMode.MAX_PERFORMANCE)
            await gopro.http_setting.max_lens_mode.set(Params.MaxLensMode.DEFAULT)
            await gopro.http_setting.camera_ux_mode.set(Params.CameraUxMode.PRO)
            await gopro.http_command.set_turbo_mode(mode=Params.Toggle.DISABLE)
            assert (await gopro.http_command.load_preset_group(group=proto.EnumPresetGroup.PRESET_GROUP_ID_VIDEO)).ok


            # Get the media list before
            media_set_before = set((await gopro.http_command.get_media_list()).data.files)
            print(media_set_before)

            # Take a video
            print("Capturing a video...")


            assert (await gopro.http_command.set_shutter(shutter=Params.Toggle.ENABLE)).ok
            await asyncio.sleep(1)
            assert (await gopro.http_command.set_shutter(shutter=Params.Toggle.DISABLE)).ok
            print("video stop")
            time.sleep(1)
            # Download all of the files from the camera

            # Get the media list after
            media_set_after = set((await gopro.http_command.get_media_list()).data.files)
            # The video (is most likely) the difference between the two sets
            video = media_set_after.difference(media_set_before).pop()
            # Download the video
            print("Downloading the video...")
            await gopro.http_command.download_file(camera_file=video.filename, local_file=video_string)
            print(f"Success!! :smiley: File has been downloaded to {video_string}")



            if gopro:
                await gopro.close()
            print("Exiting...")

        # if STREAM_PREVIEW:
        #     await gopro.http_command.set_shutter(shutter=Params.Toggle.DISABLE)
        #     await gopro.http_command.set_preview_stream(mode=Params.Toggle.DISABLE)
        #     await gopro.ble_command.set_shutter(shutter=Params.Toggle.DISABLE)
        #     assert (await gopro.http_command.set_preview_stream(mode=Params.Toggle.ENABLE)).ok
        #
        #     print("Displaying the preview stream...")
        #     display_video_blocking(r"udp://127.0.0.1:8554", printer=print)
        #     time.sleep(3)
        #
        #     await gopro.http_command.set_preview_stream(mode=Params.Toggle.DISABLE)
        if WEBCAM:
            if (await gopro.http_command.webcam_status()).data.status not in {
                WebcamStatus.OFF,
                WebcamStatus.IDLE,
            }:

                print("[blue]Webcam is currently on. Turning if off.")
                assert (await gopro.http_command.webcam_stop()).ok
                await wait_for_webcam_status(gopro, {WebcamStatus.OFF})

            print("[blue]Starting webcam...")
            await gopro.http_setting.resolution.set(Params.Resolution.RES_4K)
            await gopro.http_setting.fps.set(Params.FPS.FPS_30)
            await gopro.http_setting.video_performance_mode.set(Params.PerformanceMode.MAX_PERFORMANCE)
            await gopro.http_setting.max_lens_mode.set(Params.MaxLensMode.DEFAULT)
            await gopro.http_setting.camera_ux_mode.set(Params.CameraUxMode.PRO)
            await gopro.http_command.webcam_start()
            await wait_for_webcam_status(gopro, {WebcamStatus.HIGH_POWER_PREVIEW})

            # Start player
            display_video_blocking(STREAM_URL, printer=print)  # blocks until user exists viewer
            print("[blue]Stopping webcam...")
            assert (await gopro.http_command.webcam_stop()).ok
            await wait_for_webcam_status(gopro, {WebcamStatus.OFF, WebcamStatus.IDLE})
            assert (await gopro.http_command.webcam_exit()).ok
            await wait_for_webcam_status(gopro, {WebcamStatus.OFF})
            print("Exiting...")








if __name__ == "__main__":
    asyncio.run(main())