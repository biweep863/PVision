
void ahead(){}
void back(){}
void left(){}
void right(){}
bool pared(){}

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

// DFS algorithm
void DFS(bool visited[rows][cols], int x, int y, int directions[4][2], int rows, int cols){
    visited[x][y] = true;
    
    for (int i = 0; i<4; i++){
        int newX = x + directions[0][0];
        int newY = y + directions[0][1];
        // Serial.println(newY);
        // delay(100);
        if((newX >= 0 && newY >= 0 && newX<rows && newY <cols) && (paredAdelante() == false) && visited[newX][newY] == false){
            ahead();
            if(lineaNegra == false){
                DFS(visited, newX, newY, directions, backstep, cnt, pathFound);
                back();
            }
            else{
                visited[newX][newY] = true;
            }
        }
        girar(directions);
    }
}