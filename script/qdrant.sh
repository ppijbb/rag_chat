sudo docker pull qdrant/qdrant
sudo docker run --name qdrant_vdb -d -p 6333:6333 -p 6334:6334 -v /nas/conan/qdrant_storage:/qdrant/storage:z qdrant/qdrant
sudo docker network create qdrant_server
sudo docker network connect qdrant_server qdrant_vdb
