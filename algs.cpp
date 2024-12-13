void ahead(){}
void back(){}
void left(){}
void right(){}
bool paredAdelante(){}
bool lineaNegra = false;
// DIRECTIONS FOR UP, RIGHT, DOWN, LEFT
int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

void girar(int directions[4][2]){
    int tempx = directions[0][0];
    int tempy = directions[0][1];
    for (int i = 0; i< 3; i++){
        directions[i][0] = directions[i+1][0];
        directions[i][1] = directions[i+1][1];
    }
    directions[3][0] = tempx;
    directions[3][1] = tempy;
    left();
}

// USING DFS ALGORITHM FOR SOLVING A MID-SIZED MAZE ------------------------------------------------------
void search(bool visited[3][5], int x, int y, int directions[4][2], int backstep[3][5], int& cnt, bool& pathFound){
    visited[x][y] = true;
    if(pathFound == false){
        cnt++;
        backstep[x][y] = cnt;
    }
    if(x == 2 && y == 4){
        pathFound = true; 
    }
    for (int i = 0; i<4; i++){
        int newX = x + directions[0][0];
        int newY = y + directions[0][1];
        // Serial.println(newY);
        // delay(100);
        if((newX >= 0 && newY >= 0 && newX<3 && newY <5) && (paredAdelante() == false)){
        if(visited[newX][newY] == false){
            ahead();
            if(lineaNegra == false){
                search(visited, newX, newY, directions, backstep, cnt, pathFound);
                back();
            }
            else{
                visited[newX][newY] = true;
            }
        }
        }
        girar(directions);
    }
    if(pathFound == false){
        backstep[x][y] = 30;
        cnt--;
    }
}
void fuga(int cnt, int x, int y, int Mcolor, int directions[4][2], int backstep[3][5]){
    bool foundColor = false;
    for(int i = 1; i < cnt; i++){
        if(foundColor == false){
            if (col== Mcolor){
                Mcolor = 30;
                foundColor = true;
            }
        }
        for (int j = 0; j < 4; j++){
            int newX = x + directions[0][0];
            int newY = y + directions[0][1];
            if (newX >= 0 && newY >= 0 && newX < 3 && newY <5 && backstep[newX][newY] == i+1){
                ahead();
                j=4;
                x = newX;
                y = newY;
            }else{
                girar(directions);
            }
        }
    }
    //salir
    for(int i = 0; i< 4; i++){
        int newX = x + directions[0][0];
        int newY = y + directions[0][1];
        if(newX == 2 && newY == 5){
            ahead();
        }
}
}
void zonaC() {
    ahead();
    //adelante, derecha, atras, izquierda
    int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
    bool visited[3][5] = {{false}}; 
    int backstep[3][5] = {{30}};
    int cnt = 0;
    bool pathFound = false;
    int Mcolor = 0;
    int start_x = 1, start_y = 0;
    search(visited, start_x, start_y, directions, backstep, cnt, pathFound);
    fuga(cnt, start_x, start_y, Mcolor, directions, backstep);
}



// USING DIJKSTRA ALGORITHM FOR SOLVING A MID-SIZED MAZE ------------------------------------------------------
#include <vector>
#include <queue>
#include <climits>
#include <iostream>
using namespace std;


struct Node{
    int x, y, dist;
    bool compareNodes(const Node &a, const Node &b) {
        return a.dist > b.dist;
    }
    bool operator>(const Node &other) const{
        return dist > other.dist;
    }
};

void dijkstra(vector<vector<int>>& maze, pair<int, int> start, pair<int, int> end){
    int rows = maze.size();
    int cols = maze[0].size();

    //start matrix with infinite values
    vector<vector<int>> dist(rows, vector<int>(cols, INT_MAX));
    //start point with 0 distance
    dist[start.first][start.second] = 0;

    priority_queue<Node, vector<Node>, greater<Node>> pq;
    pq.push({start.first, start.second, 0});
    
    while(!pq.empty()){
        Node current = pq.top();
        pq.pop();
        int x = current.x, y = current.y, d = current.dist;
        if(x == end.first && y == end.second){
            cout << "Shortest distance from start to end: " << d << endl;
            return;
        }
        for(int i = 0; i<4; i++){
            int nx = x + directions[i][0];
            int ny = y + directions[i][1];
            if(nx >= 0 && ny >= 0 && nx < rows && ny < cols && maze[nx][ny] == 0){
                if(d + 1 < dist[nx][ny]){
                    dist[nx][ny] = d + 1;
                    pq.push({nx, ny, dist[nx][ny]});
                }
            }
        }
    }

}
