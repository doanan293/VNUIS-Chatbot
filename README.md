# VNUIS Chatbot
# Chatbot hỗ trợ học tập cho sinh viên Trường Quốc Tế - Đại học Quốc Gia Hà Nội 



## How to run docker container step-by-step (Ubuntu Server OS)
### 1. Cài đặt nvidia driver
```
sudo ubuntu-drivers list --gpgpu
sudo ubuntu-drivers install --gpgpu
```
### 2. Cài đặt docker 
### 3. Tạo docker image
```
sudo docker build -t tên_docker_image_muốn_đặt -f Dockerfile .
```
### 4. Chạy docker container
```
sudo docker run --network="host" tên_docker_image 
```

Hoặc nếu có CUDA chạy lệnh dưới
```
sudo docker run --gpus all --network="host" tên_docker_image
```
