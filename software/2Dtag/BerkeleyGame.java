package byow.Core;

import java.util.Random;

// This class handles running of the game via inputs, should run either using input string or typing in real time.
// Needs to provide start and end of game, and call avatar functions to allow players to play the game.

public class BerkeleyGame {
    private int width;
    private int height;
    public BerkeleyWorld berkeleyWorld;
    private CalPerson player1;
    private CalPerson player2;
    private Random rand;
    public boolean gameOver;
    private int answersToCollect;

    public BerkeleyGame(int width, int height, long seed) {
        this.width = width;
        this.height = height;

        // create map
        this.rand = new Random(seed);
        this.berkeleyWorld = new BerkeleyWorld(width, height, seed);
        this.answersToCollect = this.berkeleyWorld.rms.size() - 1;

        // set up players
        addPlayers();
    }

    public void startGame() {
        this.gameOver = false;
    }

    public void updateGame(Character c) {
        if (c.equals('l') || c.equals('L')) {
            berkeleyWorld.darkMode = !berkeleyWorld.darkMode;
        }
        if (c.equals('a') || c.equals('A')) {
            player1.moveLeft();
            berkeleyWorld.player1Position = player1.position;
        }
        if (c.equals('w') || c.equals('W')) {
            player1.moveUp();
            berkeleyWorld.player1Position = player1.position;
        }
        if (c.equals('s') || c.equals('S')) {
            player1.moveDown();
            berkeleyWorld.player1Position = player1.position;
        }
        if (c.equals('d') || c.equals('D')) {
            player1.moveRight();
            berkeleyWorld.player1Position = player1.position;
        }
        if (c.equals('4')) {
            player2.moveLeft();
        }
        if (c.equals('8')) {
            player2.moveUp();
        }
        if (c.equals('5')) {
            player2.moveDown();
        }
        if (c.equals('6')) {
            player2.moveRight();
        }
        if (berkeleyWorld.answersCollected == answersToCollect || berkeleyWorld.studentIsCaught) {
            gameOver = true;
        }
    }

    public void addPlayers() {
        boolean firstPlayerTAStatus = rand.nextBoolean();
        Position position = berkeleyWorld.POSITION;
        player1 = new CalPerson(berkeleyWorld, firstPlayerTAStatus, berkeleyWorld.getObjectLocation(position, width, height, 0,0));
        player2 = new CalPerson(berkeleyWorld, !firstPlayerTAStatus, berkeleyWorld.getObjectLocation(position, width, height,0,0));
        berkeleyWorld.changeTile(player1.position, player1.tile);
        berkeleyWorld.changeTile(player2.position, player2.tile);
        berkeleyWorld.player1Position = player1.position;
    }
}
