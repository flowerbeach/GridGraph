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

using namespace std;
long PAGESIZE = 4096;

void write_vector_edge(fstream &fout, vector<pair<VertexId, VertexId>> &vec, int start, int n) {
    for (int ii = 0; ii < n; ii++) {
        fout << vec[start + ii].first << " ";
        fout << vec[start + ii].second << " ";
    }
    fout << endl;
}

void
generate_edge_grid(const string &input, const string &output, VertexId vertices, EdgeId edges,
                   int partitions) {
    int parallelism = thread::hardware_concurrency();
    printf("vertices = %d, edges = %ld\n", vertices, edges);
    
    vector<vector<pair<VertexId, VertexId>>> buffers(parallelism * 2);
    vector<bool> occupied(parallelism * 2, false);
    Queue<tuple<int, long> > tasks(parallelism);
    vector<vector<fstream>> fout(partitions);
    if (file_exists(output)) remove_directory(output);
    create_directory(output);
    
    mutex **mutexes;
    mutexes = new mutex *[partitions];
    
    const int grid_buffer_size = 768; // 12 * 8 * 8
    vector<vector<vector<pair<VertexId, VertexId>>>> grid_buffer(partitions);
    vector<vector<int>> grid_buffer_offset(partitions);
    for (int i = 0; i < partitions; i++) {
        mutexes[i] = new mutex[partitions];
        fout[i].resize(partitions);
        grid_buffer[i].resize(partitions);
        grid_buffer_offset[i].resize(partitions);
        for (int j = 0; j < partitions; j++) {
            char filename[4096];
            sprintf(filename, "%s/block-%d-%d", output.c_str(), i, j);
            fout[i][j].open(filename, fstream::app | fstream::out);
            grid_buffer[i][j].resize(grid_buffer_size);
            grid_buffer_offset[i][j] = 0;
        }
    }
    
    vector<thread> threads;
    for (int ti = 0; ti < parallelism; ti++) {
        threads.emplace_back([&]() {
            vector<pair<VertexId, VertexId>> local_buffer;
            vector<int> local_grid_offset(partitions * partitions);
            vector<int> local_grid_cursor(partitions * partitions);
            VertexId source, target;
            Weight weight;
            while (true) {
                int cursor;
                long num_e_local;
                tie(cursor, num_e_local) = tasks.pop();
                if (cursor == -1) break;
                local_buffer.assign(num_e_local, {});
                local_grid_offset.assign(partitions * partitions, 0);
                local_grid_cursor.assign(partitions * partitions, 0);
                auto &buffer = buffers[cursor];
                for (long pos = 0; pos < num_e_local; pos++) {
                    source = buffer[pos].first;
                    target = buffer[pos].second;
                    int i = get_partition_id(vertices, partitions, source);
                    int j = get_partition_id(vertices, partitions, target);
                    local_grid_offset[i * partitions + j] += 1;
                }
                local_grid_cursor[0] = 0;
                for (int ij = 1; ij < partitions * partitions; ij++) {
                    local_grid_cursor[ij] = local_grid_offset[ij - 1];
                    local_grid_offset[ij] += local_grid_cursor[ij];
                }
                assert(local_grid_offset[partitions * partitions - 1] == num_e_local);
                for (long pos = 0; pos < num_e_local; pos++) {
                    source = buffer[pos].first;
                    target = buffer[pos].second;
                    int i = get_partition_id(vertices, partitions, source);
                    int j = get_partition_id(vertices, partitions, target);
                    local_buffer[local_grid_cursor[i * partitions + j]].first = source;
                    local_buffer[local_grid_cursor[i * partitions + j]].second = target;
                    local_grid_cursor[i * partitions + j] += 1;
                }
                int start = 0;
                for (int ij = 0; ij < partitions * partitions; ij++) {
                    assert(local_grid_cursor[ij] == local_grid_offset[ij]);
                    int i = ij / partitions;
                    int j = ij % partitions;
                    unique_lock<mutex> lock(mutexes[i][j]);
                    if (local_grid_offset[ij] - start > 1) {
                        write_vector_edge(fout[i][j], local_buffer, start, local_grid_offset[ij] - start);
                    } else if (local_grid_offset[ij] - start == 1) {
                        grid_buffer[i][j][grid_buffer_offset[i][j]].first = local_buffer[start].first;
                        grid_buffer[i][j][grid_buffer_offset[i][j]].second = local_buffer[start].second;
                        grid_buffer_offset[i][j]++;
                        if (grid_buffer_offset[i][j] == grid_buffer_size) {
                            write_vector_edge(fout[i][j], grid_buffer[i][j], 0, grid_buffer_size);
                            grid_buffer_offset[i][j] = 0;
                        }
                    }
                    start = local_grid_offset[ij];
                }
                occupied[cursor] = false;
            }
        });
    }
    
    ifstream fin;
    fin.open(input);
    if (!fin.is_open()) printf("%s\n", strerror(errno));
    int cursor = 0;
    double start_time = get_time();
    while (true) {
        long ecnt = 0;
        buffers[cursor].resize(1048576);
        while (ecnt < 1048576 && !fin.eof()) {
            fin >> buffers[cursor][ecnt].first >> buffers[cursor][ecnt].second;
            ecnt++;
        }
        if (ecnt == 0) break;
        occupied[cursor] = true;
        tasks.push(make_tuple(cursor, ecnt));
        fflush(stdout);
        while (occupied[cursor]) {
            cursor = (cursor + 1) % (parallelism * 2);
        }
    }
    fin.close();
    
    for (int ti = 0; ti < parallelism; ti++)
        tasks.push(make_tuple(-1, 0));
    for (int ti = 0; ti < parallelism; ti++)
        threads[ti].join();
    
    printf("%lf -> ", get_time() - start_time);
    long ts = 0;
    for (int i = 0; i < partitions; i++) {
        for (int j = 0; j < partitions; j++) {
            if (grid_buffer_offset[i][j] > 0) {
                ts += grid_buffer_offset[i][j];
                write_vector_edge(fout[i][j], grid_buffer[i][j], 0, grid_buffer_offset[i][j]);
            }
        }
    }
    printf("%lf (%ld)\n", get_time() - start_time, ts);
    
    for (int i = 0; i < partitions; i++) {
        for (int j = 0; j < partitions; j++) { fout[i][j].close(); }
    }
    
    printf("it takes %.2f seconds to generate edge blocks\n", get_time() - start_time);
    
    {
        char *buffer = (char *) memalign(PAGESIZE, IOSIZE);
        long offset;
        fstream fout_column, fout_column_offset;
        fout_column.open((output + "/column").c_str(), fstream::app | fstream::out);
        fout_column_offset.open((output + "/column_offset").c_str(), fstream::app | fstream::out);
        offset = 0;
        for (int j = 0; j < partitions; j++) {
            for (int i = 0; i < partitions; i++) {
                fflush(stdout);
                fout_column_offset << offset << " ";
                char filename[4096];
                sprintf(filename, "%s/block-%d-%d", output.c_str(), i, j);
                offset += file_size(filename);
                fin.open(filename);
                while (true) {
                    fin.read(buffer, IOSIZE);
                    long bytes = fin.gcount();
                    assert(bytes != -1);
                    if (bytes == 0) break;
                    fout_column.write(buffer, bytes);
                }
                fin.close();
            }
        }
        fout_column_offset << offset << " ";
        fout_column_offset.close();
        fout_column.close();
        printf("column oriented grid generated\n");
        
        fstream fout_row, fout_row_offset;
        fout_row.open((output + "/row").c_str(), fstream::app | fstream::out);
        fout_row_offset.open((output + "/row_offset").c_str(), fstream::app | fstream::out);
        offset = 0;
        for (int i = 0; i < partitions; i++) {
            for (int j = 0; j < partitions; j++) {
                fflush(stdout);
                fout_row_offset << offset << " ";
                char filename[4096];
                sprintf(filename, "%s/block-%d-%d", output.c_str(), i, j);
                offset += file_size(filename);
                fin.open(filename);
                while (true) {
                    fin.read(buffer, IOSIZE);
                    long bytes = fin.gcount();
                    assert(bytes != -1);
                    if (bytes == 0) break;
                    fout_row.write(buffer, bytes);
                }
                fin.close();
            }
        }
        fout_row_offset << offset << " ";
        fout_row_offset.close();
        fout_row.close();
        printf("row oriented grid generated\n");
    }
    
    printf("it takes %.2f seconds to generate edge grid\n", get_time() - start_time);
    
    FILE *fmeta = fopen((output + "/meta").c_str(), "w");
    fprintf(fmeta, "%d %ld %d", vertices, edges, partitions);
    fclose(fmeta);
}

int main(int argc, char **argv) {
    int opt;
    string input;
    string output;
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
    if (input.empty() || output.empty() || vertices == -1) {
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
