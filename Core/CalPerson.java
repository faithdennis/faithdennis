package byow.Core;

import byow.TileEngine.TETile;
import byow.TileEngine.Tileset;

// This class handles avatar related functions, place any functions specific to the avatars here including checking type,
// grabbing documents, or moving.

public class CalPerson {
    boolean isTA;
    Position position;
    BerkeleyWorld berkeleyWorld;
    TETile[][] campus;
    TETile tile;
    CalPerson(BerkeleyWorld berkeleyWorld, boolean TA, Position position) {
            this.isTA = TA;
            this.position = position;
            this.berkeleyWorld = berkeleyWorld;
            this.campus = this.berkeleyWorld.getCampus();

            if (TA) {
                this.tile = Tileset.TA;
            } else { this.tile = Tileset.STUDENT; }

            this.berkeleyWorld.changeTile(this.position, this.tile);
    }

    public void moveLeft() {
        Position next = position.shift(- 1, 0);
        if (berkeleyWorld.hasFloor(next)) {
            berkeleyWorld.changeTile(position, Tileset.FLOOR);
            berkeleyWorld.changeTile(next, tile);
            position = next;
        }
        if (berkeleyWorld.hasStudent(next) && isTA) {
            berkeleyWorld.changeTile(next, Tileset.CONVICT);
            berkeleyWorld.studentIsCaught = true;
        }
        if (berkeleyWorld.hasAnswerSheet(next) && !isTA) {
            collectAnswers(next);
        }
    }

    public void moveUp() {
        Position next = position.shift(0, 1);
        if (berkeleyWorld.hasFloor(next)) {
            berkeleyWorld.changeTile(position, Tileset.FLOOR);
            berkeleyWorld.changeTile(next, tile);
            position = next;
        }
        if (berkeleyWorld.hasStudent(next) && isTA) {
            berkeleyWorld.changeTile(next, Tileset.CONVICT);
            berkeleyWorld.studentIsCaught = true;
        }
        if (berkeleyWorld.hasAnswerSheet(next) && !isTA) {
            collectAnswers(next);
        }
    }

    public void moveDown() {
        Position next = position.shift(0, -1);
        if (berkeleyWorld.hasFloor(next)) {
            berkeleyWorld.changeTile(position, Tileset.FLOOR);
            berkeleyWorld.changeTile(next, tile);
            position = next;
        }
        if (berkeleyWorld.hasStudent(next) && isTA) {
            berkeleyWorld.changeTile(next, Tileset.CONVICT);
            berkeleyWorld.studentIsCaught = true;
        }
        if (berkeleyWorld.hasAnswerSheet(next) && !isTA) {
            collectAnswers(next);
        }
    }

    public void moveRight() {
        Position next = position.shift(1, 0);
        if (berkeleyWorld.hasFloor(next)) {
            berkeleyWorld.changeTile(position, Tileset.FLOOR);
            berkeleyWorld.changeTile(next, tile);
            position = next;
        }
        if (berkeleyWorld.hasStudent(next) && isTA) {
            berkeleyWorld.changeTile(next, Tileset.CONVICT);
            berkeleyWorld.studentIsCaught = true;
        }
        if (berkeleyWorld.hasAnswerSheet(next) && !isTA) {
            collectAnswers(next);
        }
    }

    public void collectAnswers(Position p) {
        if (isTA) {
            throw new UnsupportedOperationException("TA's should not collect answers~");
        }
        berkeleyWorld.changeTile(p, tile);
        berkeleyWorld.changeTile(position, Tileset.FLOOR);
        position = p;
        berkeleyWorld.answersCollected++;
    }
}
