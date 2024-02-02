import cv2
import os
import subprocess

import open3d as o3d


def extract_frames(video_path, output_folder, record_file):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the record file to get a list of already extracted frames
    if os.path.exists(record_file):
        with open(record_file, 'r') as f:
            extracted_frames = [line.strip() for line in f.readlines()]
    else:
        extracted_frames = []

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")

        # Check if the frame has already been extracted
        if frame_filename in extracted_frames:
            print(f"Skipping already extracted frame: {frame_filename}")
            continue

        # Save the frame
        cv2.imwrite(frame_filename, frame)
        print(f"Extracted frame: {frame_filename}")

        # Record the filename in the record file
        with open(record_file, 'a') as record:
            record.write(frame_filename + '\n')

    cap.release()


def run_micmac(input_folder, output_folder):
    print("micmac")
    mm3d_directory ="/home/mmt-ben/micmac/bin/"

    #"mm3d C3DC /home/mmt-ben/MAPPER_AGRI_MULTICAM/photogrammetry_images/*.jpg"

    command = "mm3d C3DC " + input_folder + "*.jpg"
    print(command)
    micmac_command = os.path.join(mm3d_directory,
                                  command)

    subprocess.run(micmac_command, check=True, shell=True, cwd=output_folder)

def generate_point_cloud(output_folder):
    print("generate_point_cloud")
    point_cloud_command = "mm3d Malt"
    subprocess.run(point_cloud_command, check=True, shell=True, cwd=output_folder)



def export_point_cloud(output_folder, output_ply):
    print("export_point_cloud")

    export_command = f"mm3d AperiCloud {output_folder}/Res/OutP.ply {output_ply}"
    subprocess.run(export_command, check=True, shell=True)

if __name__ == "__main__":
    # Example usage
    video_path = "/home/mmt-ben/MAPPER_AGRI_MULTICAM/aquisition/GX010066.MP4"
    output_folder = "/home/mmt-ben/MAPPER_AGRI_MULTICAM/photogrammetry_images/"
    pointcloud_folder = "/home/mmt-ben/MAPPER_AGRI_MULTICAM/pointcloud"
    output_ply = "/home/mmt-ben/MAPPER_AGRI_MULTICAM/pointcloud/output_point_cloud.ply"
    record_file = "/home/mmt-ben/MAPPER_AGRI_MULTICAM/record.txt"

    #extract_frames(video_path, output_folder,record_file)


    # Example usage
    run_micmac(output_folder, pointcloud_folder)


    # Example usage
    generate_point_cloud(output_folder)


    # Example usage

    export_point_cloud(output_folder, output_ply)


    # Load the point cloud
    point_cloud = o3d.io.read_point_cloud(output_ply)
    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])
