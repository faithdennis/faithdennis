package byow.Core;

import byow.TileEngine.TETile;
import byow.TileEngine.Tileset;

import java.util.ArrayList;
import java.util.Random;

// BerkeleyWorld extends campus, campus builds a map whereas BerkeleyWorld adds gameplay mechanics on top of that.
// In this case BerkeleyWorld allows us to change tiles, get object locations, and check whether certain tiles contain certain objects.
// Put any map changing or checking functions here needed for the game.

public class BerkeleyWorld extends Campus {
    int answersCollected;
    boolean studentIsCaught;
    boolean darkMode;
    Position player1Position;
    TETile[][] darkCampus = new TETile[WIDTH][HEIGHT];

    BerkeleyWorld(int width, int height, long seed) {
        super(width, height, seed);
        player1Position = POSITION;
        darkMode = false;
        answersCollected = 0;
        studentIsCaught = false;
        placeDocuments();
    }

    @Override
    public TETile[][] getCampus() {
        if (darkMode) {
            createDarkCampus();
            return darkCampus;
        }
        else {
            return campus;
        }
    }

    public TETile[][] createDarkCampus() {
        Position left = player1Position.shift(-2,0);
        Position right = player1Position.shift(2,0);
        Position top = player1Position.shift(0,2);
        Position bottom = player1Position.shift(0, -2);
        for (int x = 0; x < WIDTH; x++) {
            for (int y = 0; y < HEIGHT; y++) {
                if (x >= left.x && x <= right.x && y >= bottom.y && y <= top.y) {
                    darkCampus[x][y] = campus[x][y];
                }
                else { darkCampus[x][y] = Tileset.NOTHING; }
            }
        }
        return darkCampus;
    }

    public Position getObjectLocation(Position startofPart, int width, int height, int bottom, int top) {
        Position random = startofPart;
        while (campus[random.x][random.y] != Tileset.FLOOR) {
            random = getRandomPosition(startofPart, width, height, bottom, top);
        }
        return random;
    }

    public void changeTile(Position p, TETile tile) {
        campus[p.x][p.y] = tile;
    }

    public boolean hasStudent(Position p) {
        return campus[p.x][p.y] == Tileset.STUDENT;
    }

    public boolean hasAnswerSheet(Position p) {
        return campus[p.x][p.y] == Tileset.ANS;
    }

    public boolean hasFloor(Position p) {
        return campus[p.x][p.y] == Tileset.FLOOR;
    }

    public void placeDocuments() {
        for (Room rm : rms) {
            changeTile(rm.midpoint, Tileset.ANS);
        }
    }
}
