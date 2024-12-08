#include <iostream>
#include <vector>
#include <queue>
#include <climits>

using namespace std;

// Directions: right, down, left, up
const int dx[] = {0, 1, 0, -1};
const int dy[] = {1, 0, -1, 0};

// Struct for the priority queue (min-heap)
struct Node {
    int x, y, dist;
    bool operator>(const Node &other) const {
        return dist > other.dist;  // For min-heap based on distance
    }
};

// Dijkstra's Algorithm for solving the maze
void dijkstra(vector<vector<int>>& maze, pair<int, int> start, pair<int, int> end) {
    int rows = maze.size();
    int cols = maze[0].size();

    // Distance matrix initialized to infinity
    vector<vector<int>> dist(rows, vector<int>(cols, INT_MAX));
    dist[start.first][start.second] = 0;

    // Min-heap (priority queue)
    priority_queue<Node, vector<Node>, greater<Node>> pq;
    pq.push({start.first, start.second, 0});

    while (!pq.empty()) {
        Node current = pq.top();
        pq.pop();

        int x = current.x, y = current.y, d = current.dist;

        // If we reached the destination, print the distance and exit
        if (x == end.first && y == end.second) {
            cout << "Shortest distance from start to end: " << d << endl;
            return;
        }

        // Explore the four possible directions (right, down, left, up)
        for (int i = 0; i < 4; ++i) {
            int nx = x + dx[i];
            int ny = y + dy[i];

            // Check bounds and if the cell is open (0)
            if (nx >= 0 && ny >= 0 && nx < rows && ny < cols && maze[nx][ny] == 0) {
                // If a shorter path is found
                if (d + 1 < dist[nx][ny]) {
                    dist[nx][ny] = d + 1;
                    pq.push({nx, ny, dist[nx][ny]});
                }
            }
        }
    }

    cout << "No path found from start to end." << endl;
}

int main() {
    // Define the maze: 0 is free, 1 is wall
    vector<vector<int>> maze = {
        {0, 1, 0, 0, 0},
        {0, 1, 0, 1, 0},
        {0, 0, 0, 1, 0},
        {1, 1, 0, 0, 0},
        {0, 0, 0, 1, 0}
    };

    pair<int, int> start = {0, 0};  // Start point (top-left corner)
    pair<int, int> end = {4, 4};    // End point (bottom-right corner)

    // Run Dijkstra's algorithm
    dijkstra(maze, start, end);
    return 0;
}
