/*
Copyright (c) 2014-2015 Xiaowei Zhu, Tsinghua University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <malloc.h>
#include <cerrno>
#include <cassert>
#include <cstring>
#include <fstream>

#include <string>
#include <vector>
#include <thread>

#include "../core/constants.hpp"
#include "../core/type.hpp"
#include "../core/filesystem.hpp"
#include "../core/queue.hpp"
#include "../core/partition.hpp"
#include "../core/time.hpp"
#include "../core/atomic.hpp"

long PAGESIZE = 4096;

void
generate_edge_grid(std::string input, std::string output, VertexId vertices, EdgeId edges, int partitions) {
    int parallelism = std::thread::hardware_concurrency();
    int edge_unit;
    printf("vertices = %d, edges = %ld\n", vertices, edges);
    
    char **buffers = new char *[parallelism * 2];
    bool *occupied = new bool[parallelism * 2];
    for (int i = 0; i < parallelism * 2; i++) {
        buffers[i] = (char *) memalign(PAGESIZE, IOSIZE);
        occupied[i] = false;
    }
    Queue<std::tuple<int, long> > tasks(parallelism);
    std::fstream **fout;
    std::mutex **mutexes;
    fout = new std::fstream *[partitions];
    mutexes = new std::mutex *[partitions];
    if (file_exists(output)) {
        remove_directory(output);
    }
    create_directory(output);
    
    const int grid_buffer_size = 768; // 12 * 8 * 8
    char *global_grid_buffer = (char *) memalign(PAGESIZE, grid_buffer_size * partitions * partitions);
    char ***grid_buffer = new char **[partitions];
    int **grid_buffer_offset = new int *[partitions];
    for (int i = 0; i < partitions; i++) {
        mutexes[i] = new std::mutex[partitions];
        fout[i] = new std::fstream[partitions];
        grid_buffer[i] = new char *[partitions];
        grid_buffer_offset[i] = new int[partitions];
        for (int j = 0; j < partitions; j++) {
            char filename[4096];
            sprintf(filename, "%s/block-%d-%d", output.c_str(), i, j);
            fout[i][j].open(filename, std::fstream::app | std::fstream::out);
            grid_buffer[i][j] = global_grid_buffer + (i * partitions + j) * grid_buffer_size;
            grid_buffer_offset[i][j] = 0;
        }
    }
    
    std::vector<std::thread> threads;
    for (int ti = 0; ti < parallelism; ti++) {
        threads.emplace_back([&]() {
            char *local_buffer = (char *) memalign(PAGESIZE, IOSIZE);
            int *local_grid_offset = new int[partitions * partitions];
            int *local_grid_cursor = new int[partitions * partitions];
            VertexId source, target;
            Weight weight;
            while (true) {
                int cursor;
                long bytes;
                std::tie(cursor, bytes) = tasks.pop();
                if (cursor == -1) break;
                memset(local_grid_offset, 0, sizeof(int) * partitions * partitions);
                memset(local_grid_cursor, 0, sizeof(int) * partitions * partitions);
                char *buffer = buffers[cursor];
                for (long pos = 0; pos < bytes; pos += edge_unit) {
                    source = *(VertexId *) (buffer + pos);
                    target = *(VertexId *) (buffer + pos + sizeof(VertexId));
                    int i = get_partition_id(vertices, partitions, source);
                    int j = get_partition_id(vertices, partitions, target);
                    local_grid_offset[i * partitions + j] += edge_unit;
                }
                local_grid_cursor[0] = 0;
                for (int ij = 1; ij < partitions * partitions; ij++) {
                    local_grid_cursor[ij] = local_grid_offset[ij - 1];
                    local_grid_offset[ij] += local_grid_cursor[ij];
                }
                assert(local_grid_offset[partitions * partitions - 1] == bytes);
                for (long pos = 0; pos < bytes; pos += edge_unit) {
                    source = *(VertexId *) (buffer + pos);
                    target = *(VertexId *) (buffer + pos + sizeof(VertexId));
                    int i = get_partition_id(vertices, partitions, source);
                    int j = get_partition_id(vertices, partitions, target);
                    *(VertexId *) (local_buffer + local_grid_cursor[i * partitions + j]) = source;
                    *(VertexId *) (local_buffer + local_grid_cursor[i * partitions + j] + sizeof(VertexId)) = target;
                    local_grid_cursor[i * partitions + j] += edge_unit;
                }
                int start = 0;
                for (int ij = 0; ij < partitions * partitions; ij++) {
                    assert(local_grid_cursor[ij] == local_grid_offset[ij]);
                    int i = ij / partitions;
                    int j = ij % partitions;
                    std::unique_lock<std::mutex> lock(mutexes[i][j]);
                    if (local_grid_offset[ij] - start > edge_unit) {
                        fout[i][j].write(local_buffer + start, local_grid_offset[ij] - start);
                    } else if (local_grid_offset[ij] - start == edge_unit) {
                        memcpy(grid_buffer[i][j] + grid_buffer_offset[i][j], local_buffer + start, edge_unit);
                        grid_buffer_offset[i][j] += edge_unit;
                        if (grid_buffer_offset[i][j] == grid_buffer_size) {
                            fout[i][j].write(grid_buffer[i][j], grid_buffer_size);
                            grid_buffer_offset[i][j] = 0;
                        }
                    }
                    start = local_grid_offset[ij];
                }
                occupied[cursor] = false;
            }
        });
    }
    
    std::ifstream fin;
    if (!fin.is_open()) printf("%s\n", strerror(errno));
    int cursor = 0;
    double start_time = get_time();
    while (true) {
        fin.read(buffers[cursor], IOSIZE);
        long bytes = fin.gcount();
        assert(bytes != -1);
        if (bytes == 0) break;
        occupied[cursor] = true;
        tasks.push(std::make_tuple(cursor, bytes));
        fflush(stdout);
        while (occupied[cursor]) {
            cursor = (cursor + 1) % (parallelism * 2);
        }
    }
    fin.close();
    
    for (int ti = 0; ti < parallelism; ti++) {
        tasks.push(std::make_tuple(-1, 0));
    }
    
    for (int ti = 0; ti < parallelism; ti++) {
        threads[ti].join();
    }
    
    printf("%lf -> ", get_time() - start_time);
    long ts = 0;
    for (int i = 0; i < partitions; i++) {
        for (int j = 0; j < partitions; j++) {
            if (grid_buffer_offset[i][j] > 0) {
                ts += grid_buffer_offset[i][j];
                fout[i][j].write(grid_buffer[i][j], grid_buffer_offset[i][j]);
            }
        }
    }
    printf("%lf (%ld)\n", get_time() - start_time, ts);
    
    for (int i = 0; i < partitions; i++) {
        for (int j = 0; j < partitions; j++) {
            fout[i][j].close();
        }
    }
    
    printf("it takes %.2f seconds to generate edge blocks\n", get_time() - start_time);
    
    long offset;
    std::fstream fout_column;
    fout_column.open((output + "/column").c_str(), std::fstream::app | std::fstream::out);
    std::fstream fout_column_offset;
    fout_column_offset.open((output + "/column_offset").c_str(), std::fstream::app | std::fstream::out);
    offset = 0;
    for (int j = 0; j < partitions; j++) {
        for (int i = 0; i < partitions; i++) {
            fflush(stdout);
            fout_column_offset.write(std::to_string(offset).c_str(), sizeof(offset));
            char filename[4096];
            sprintf(filename, "%s/block-%d-%d", output.c_str(), i, j);
            offset += file_size(filename);
            fin.open(filename);
            while (true) {
                fin.read(buffers[0], IOSIZE);
                long bytes = fin.gcount();
                assert(bytes != -1);
                if (bytes == 0) break;
                fout_column.write(buffers[0], bytes);
            }
            fin.close();
        }
    }
    fout_column_offset.write(std::to_string(offset).c_str(), sizeof(offset));
    fout_column_offset.close();
    fout_column.close();
    printf("column oriented grid generated\n");
    std::fstream fout_row;
    fout_row.open((output + "/row").c_str(), std::fstream::app | std::fstream::out);
    std::fstream fout_row_offset;
    fout_row_offset.open((output + "/row_offset").c_str(), std::fstream::app | std::fstream::out);
    offset = 0;
    for (int i = 0; i < partitions; i++) {
        for (int j = 0; j < partitions; j++) {
            fflush(stdout);
            fout_row_offset.write(std::to_string(offset).c_str(), sizeof(offset));
            char filename[4096];
            sprintf(filename, "%s/block-%d-%d", output.c_str(), i, j);
            offset += file_size(filename);
            fin.open(filename);
            while (true) {
                fin.read(buffers[0], IOSIZE);
                long bytes = fin.gcount();
                assert(bytes != -1);
                if (bytes == 0) break;
                fout_row.write(buffers[0], bytes);
            }
            fin.close();
        }
    }
    fout_row_offset.write(std::to_string(offset).c_str(), sizeof(offset));
    fout_row_offset.close();
    fout_row.close();
    printf("row oriented grid generated\n");
    
    printf("it takes %.2f seconds to generate edge grid\n", get_time() - start_time);
    
    FILE *fmeta = fopen((output + "/meta").c_str(), "w");
    fprintf(fmeta, "%d %ld %d", vertices, edges, partitions);
    fclose(fmeta);
}

int main(int argc, char **argv) {
    int opt;
    std::string input = "";
    std::string output = "";
    EdgeId edges = 68993773;
    VertexId vertices = -1;
    int partitions = -1;
    int edge_type = 0;
    while ((opt = getopt(argc, argv, "i:o:v:p:t:")) != -1) {
        switch (opt) {
            case 'i':
                input = optarg;
                break;
            case 'o':
                output = optarg;
                break;
            case 'v':
                vertices = atoi(optarg);
                break;
            case 'p':
                partitions = atoi(optarg);
                break;
            default:
                break;
        }
    }
    if (input == "" || output == "" || vertices == -1) {
        fprintf(stderr,
                "usage: %s -i [input path] -o [output path] -v [vertices] -p [partitions] -t [edge type: 0=unweighted, 1=weighted]\n",
                argv[0]);
        exit(-1);
    }
    if (partitions == -1) {
        partitions = vertices / CHUNKSIZE;
    }
    generate_edge_grid(input, output, vertices, edges, partitions);
    return 0;
}
