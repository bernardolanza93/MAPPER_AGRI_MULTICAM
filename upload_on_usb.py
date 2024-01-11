import os
import shutil


def find_usb_drive():
    # List all devices mounted under /media
    username = os.getlogin()
    print(f"The current username is: {username}")
    path_usb = '/media/'+str(username)
    print("search in",path_usb)
    mounted_devices = os.listdir(path_usb)


    # Check for devices containing 'usb' in the name
    usb_drives = [device for device in mounted_devices]
    print("usb:",usb_drives)

    if usb_drives:
        return path_usb + '/'+usb_drives[0]  # Return the path of the first found USB drive
    else:
        print("NO USB DETECTED")
        return None  # Return None if no USB drive is found


def copy_to_usb(source_folder, destination_folder):
    usb_path = find_usb_drive()
    print("FOUND PENDRIVE:",usb_path)

    if usb_path:
        source_path = os.path.abspath(source_folder)
        destination_path = os.path.join(usb_path, destination_folder)

        # Check if the destination folder already exists
        if os.path.exists(destination_path):
            try:
                # Remove the existing destination folder and its contents
                shutil.rmtree(destination_path)
                print("Existing destination folder removed.")
            except Exception as e:
                print(f"Error removing existing folder: {e}")

        try:
            # Copy the entire directory tree (files and folders) to the USB drive
            shutil.copytree(source_path, destination_path)
            print("Files and folders copied successfully to USB drive.")
            print(source_path)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("USB drive not found.")


# Replace 'source_folder' and 'destination_folder' with your actual source and destination paths
copy_to_usb('aquisition_raw', 'data_from_jetson')
