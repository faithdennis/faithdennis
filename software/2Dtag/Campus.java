package byow.Core;

import byow.TileEngine.TETile;
import byow.TileEngine.Tileset;

import java.util.ArrayList;
import java.util.Random;

public class Campus {
    protected int WIDTH;
    protected int HEIGHT;
    protected final int MINROOMSIZE = 4;
    protected final int MAXROOMSIZE = 18;
    protected final Position POSITION = new Position(0,0);
    protected final int PARTITIONNUM = 3;
    protected Random RANDOM;
    protected TETile[][] campus;
    protected ArrayList<Room> rms = new ArrayList<>();

    Campus(int width, int height, long seed) {
        this.WIDTH = width;
        this.HEIGHT = height;
        this.RANDOM = new Random(seed);
        campus = new TETile[WIDTH][HEIGHT];
        makeCampusGreen(campus);
        buildRoomsOnCampus(campus, POSITION, WIDTH, HEIGHT, PARTITIONNUM);
        buildHallWays(campus, rms);
    }

    public TETile[][] getCampus() {
        return campus;
    }

    public void makeCampusGreen(TETile[][] world) {
        for (int x = 0; x < WIDTH; x++) {
            for (int y = 0; y < HEIGHT; y++) {
                world[x][y] = Tileset.GRASS2;
            }
        }
    }

    public Position getRandomPosition(Position startofPart, int width, int height, int bottom, int top) {
        Position endofPart = startofPart.shift(width,height);
        int bx = startofPart.x + bottom + 1;
        int tx = endofPart.x - top;
        int by = startofPart.y + bottom + 1;
        int ty = endofPart.y - top;
        int xPosition;
        int yPosition;

        if (bx == tx) {
            xPosition = bx;
        } else {
            xPosition = RANDOM.nextInt(bx, tx);
        }
        if (by == ty) {
            yPosition = by;
        } else {
            yPosition = RANDOM.nextInt(by, ty);
        }
        return new Position(xPosition, yPosition);
    }

    public Position partition(Position start, Position location, int type) {
        if (type == 0) {
            return new Position(start.x, location.y);
        }
        else {
            return new Position(location.x, start.y);
        }
    }


    protected class Room {
        Position loc;
        Position midpoint;
        Position doorway;
        int w;
        int h;

        Room(Position p, int w, int h) {
            this.w = w;
            this.h = h;
            this.loc = p;
            this.midpoint = p.shift((int) Math.floor(w/2), (int) Math.floor(h/2));
            this.doorway = getRandomPosition(p, w, h, 1,1);
        }
    }

    public void buildRoomsOnCampus(TETile[][] world, Position p, int width, int height, int partitionNum) {
        int partitionCount = partitionNum;

        if (partitionCount > 0) {
            int partitionType = RANDOM.nextInt(0, 2); // pick partition direction randomly
            Position partitionAt = getRandomPosition(p, width, height, partitionCount * MINROOMSIZE, partitionCount * MINROOMSIZE); // choose random position
            partitionCount-- ; // decrement bc we created a partition

            Position secondPartition = partition(p, partitionAt, partitionType); // position of new partition
            int secondWidth = width - (secondPartition.x - p.x);
            int secondHeight = height - (secondPartition.y - p.y);
            int firstWidth = width - secondWidth;
            int firstHeight = height - secondHeight;

            if (partitionType == 0) {
                firstWidth = width;
            }
            if (partitionType == 1) {
                firstHeight = height;
            }

            buildRoomsOnCampus(world, p, firstWidth, firstHeight, partitionCount);
            buildRoomsOnCampus(world, secondPartition, secondWidth, secondHeight, partitionCount);
        }

        if (partitionNum == 0) {
            if (width <= MINROOMSIZE || height <= MINROOMSIZE) {
                return;
            }
            Room rm = createRoom(world, p, width, height);
            rms.add(rm);
        }
    }

    public void buildHallWays(TETile[][] tiles, ArrayList<Room> lst) {
        for (int i = 0; i < lst.size(); i++) {
            for (int j = i + 1; j < lst.size(); j++) {
                int direction = RANDOM.nextInt(2);
                Position startPosition = lst.get(i).doorway;
                Position endPosition = lst.get(j).doorway;

                hallwayHelper(tiles, startPosition, endPosition, direction);
            }
        }
    }

    public Room createRoom(TETile[][] world, Position p, int w, int h) {
        int maxW = Integer.min(MAXROOMSIZE, w);
        int maxH = Integer.min(MAXROOMSIZE, h);
        int randomWidth = RANDOM.nextInt(MINROOMSIZE, maxW);
        int randomHeight = RANDOM.nextInt(MINROOMSIZE, maxH);
        Position startofRoom = new Position(RANDOM.nextInt(p.x, p.x + w - randomWidth), RANDOM.nextInt(p.y,p.y + h - randomHeight));
        Position endofRoom = startofRoom.shift(randomWidth, randomHeight);


        for (int x = startofRoom.x; x <= endofRoom.x; x++) {
            for (int y = startofRoom.y; y <= endofRoom.y; y++) {
                if (x == startofRoom.x || x == endofRoom.x || y == startofRoom.y || y == endofRoom.y) {
                    world[x][y] = Tileset.WALL;
                } else {
                    world[x][y] = Tileset.FLOOR;
                }
            }
        }
        return new Room(startofRoom, randomWidth, randomHeight);
    }

    public void hallwayHelper(TETile[][] tiles, Position start, Position end, int direction) {
        direction = 1 - direction;

        if (start.x == end.x && start.y == end.y) {
            return;
        }
        else if (direction == 0) {
            Position newPosition = horizontalHallWay(tiles, start, end);
            hallwayHelper(tiles, newPosition, end, direction);
        }
        else if (direction == 1) {
            Position newPosition = verticalHallWay(tiles, start, end);
            hallwayHelper(tiles, newPosition, end, direction);
        }
    }

    public Position verticalHallWay(TETile[][] tiles, Position start, Position end) {
        int startY = Integer.min(start.y, end.y);
        int endY = Integer.max(start.y, end.y);
        int x = start.x;

        for (int y = startY; y <= endY; y++) {
            if (x == 0 || x == WIDTH - 1 || y == 0 || y == HEIGHT - 1) {
                tiles[x][y] = Tileset.WALL;
            }
            else if (tiles[x][y] == Tileset.GRASS2 || tiles[x][y] == Tileset.WALL) {
                tiles[x][y] = Tileset.FLOOR;
                if (tiles[x - 1][y] == Tileset.GRASS2) {
                    tiles[x - 1][y] = Tileset.WALL;
                }
                if (tiles[x + 1][y] == Tileset.GRASS2) {
                    tiles[x + 1][y] = Tileset.WALL;
                }
                if (y == endY) {
                    if (tiles[x - 1][y + 1] == Tileset.GRASS2) {
                        tiles[x - 1][y + 1] = Tileset.WALL;
                    }
                    if (tiles[x][y + 1] == Tileset.GRASS2) {
                        tiles[x][y + 1] = Tileset.WALL;
                    }
                    if (tiles[x + 1][y + 1] == Tileset.GRASS2) {
                        tiles[x + 1][y + 1] = Tileset.WALL;
                    }
                }
                if (y == startY) {
                    if (tiles[x - 1][y - 1] == Tileset.GRASS2) {
                        tiles[x - 1][y - 1] = Tileset.WALL;
                    }
                    if (tiles[x][y - 1] == Tileset.GRASS2) {
                        tiles[x][y - 1] = Tileset.WALL;
                    }
                    if (tiles[x + 1][y - 1] == Tileset.GRASS2) {
                        tiles[x + 1][y - 1] = Tileset.WALL;
                    }
                }
            }
        }
        return new Position(x, end.y);
    }

    public Position horizontalHallWay(TETile[][] tiles, Position start, Position end) {
        int startX = Integer.min(start.x, end.x);
        int endX = Integer.max(start.x, end.x);
        int y = start.y;
        for (int x = startX; x <= endX; x++) {
            if (x == 0 || x == WIDTH - 1 || y == 0 || y == HEIGHT - 1) {
                tiles[x][y] = Tileset.WALL;
            }
            else if (tiles[x][y] == Tileset.GRASS2 || tiles[x][y] == Tileset.WALL) {
                tiles[x][y] = Tileset.FLOOR;
                if (tiles[x][y - 1] == Tileset.GRASS2) {
                    tiles[x][y - 1] = Tileset.WALL;
                }
                if (tiles[x][y + 1] == Tileset.GRASS2) {
                    tiles[x][y + 1] = Tileset.WALL;
                }
                if (x == endX) {
                    if (tiles[x + 1][y - 1] == Tileset.GRASS2) {
                        tiles[x + 1][y - 1] = Tileset.WALL;
                    }
                    if (tiles[x + 1][y] == Tileset.GRASS2) {
                        tiles[x + 1][y] = Tileset.WALL;
                    }
                    if (tiles[x + 1][y + 1] == Tileset.GRASS2) {
                        tiles[x + 1][y + 1] = Tileset.WALL;
                    }
                }
                if (x == startX) {
                    if (tiles[x - 1][y - 1] == Tileset.GRASS2) {
                        tiles[x - 1][y - 1] = Tileset.WALL;
                    }
                    if (tiles[x - 1][y] == Tileset.GRASS2) {
                        tiles[x - 1][y] = Tileset.WALL;
                    }
                    if (tiles[x - 1][y + 1] == Tileset.GRASS2) {
                        tiles[x - 1][y + 1] = Tileset.WALL;
                    }
                }
            }
        }
        return new Position(end.x, start.y);
    }
}

