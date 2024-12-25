package byow.Core;

import byow.TileEngine.TERenderer;
import byow.TileEngine.TETile;
import edu.princeton.cs.algs4.StdDraw;

import java.awt.*;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class Engine {
    TERenderer ter = new TERenderer();
    public static final int WIDTH = 90;
    public static final int HEIGHT = 45;
    public static final int CANVASW = WIDTH * 16;
    public static final int CANVASH = HEIGHT * 16;

    /**
     * Method used for exploring a fresh world. This method should handle all inputs,
     * including inputs from the main menu.
     */
    public void interactWithKeyboard() {
        String seed = "";
        String data = "";
        Character curr = ' ';
        String menuOptions = "NnLlIiQq";

        // sets up menu
        drawMenu();

        // waiting for user to start a valid command
        while (!menuOptions.contains(Character.toString(curr))) {
            if (StdDraw.hasNextKeyTyped()) {
                curr = StdDraw.nextKeyTyped();
            }
        }

        // user initiates a new game
        if (curr.equals('N') || curr.equals('n')) {
            while (!curr.equals('S') && !curr.equals('s')) {
                showSeed(seed);
                if (StdDraw.hasNextKeyTyped()) {
                    curr = StdDraw.nextKeyTyped();
                    if (Character.isDigit(curr)) {
                        seed += curr;
                    }
                }
            }
            // records seed data
            data = 'n' + seed + 's';

            // starts game
            ter.initialize(WIDTH, HEIGHT);
            BerkeleyGame bg = new BerkeleyGame(90, 45, Integer.parseInt(seed));
            bg.startGame();
            ter.renderFrame(bg.berkeleyWorld.getCampus());

            // plays game
            playGame(bg, data);
        }

        if (curr.equals('L') || curr.equals('l')) {
            data = loadFile();
            loadGame(data);
        }

        if (curr.equals('Q') || curr.equals('q')) {
            System.exit(0);
        }
    }

    public void drawMenu() {
        /* Take the input string S and display it at the center of the screen,
         * with the pen settings given below. */
        StdDraw.setCanvasSize(CANVASW, CANVASH);
        StdDraw.clear(Color.BLACK);
        StdDraw.setPenColor(Color.WHITE);
        Font font = new Font("Monaco", Font.BOLD, 30);
        StdDraw.setFont(font);
        StdDraw.setXscale(0, WIDTH);
        StdDraw.setYscale(0, HEIGHT);
        StdDraw.enableDoubleBuffering();

        StdDraw.text(WIDTH / 2, (HEIGHT / 2) + 10, "berkeleyWorld");
        StdDraw.text(WIDTH / 2, HEIGHT / 2, "new world (n)");
        StdDraw.text(WIDTH / 2, (HEIGHT / 2) - 5, "load (l)");
        StdDraw.text(WIDTH / 2, (HEIGHT / 2) - 10, "quit (q)");

        //TODO: If the game is not over, display encouragement and let the user know if they
        // should be typing their answer or watching for the next round.
        // Font fontMedium = new Font("Monaco", Font.BOLD, 20);
        // StdDraw.setFont(fontMedium);
        // StdDraw.line(0, HEIGHT - 2, WIDTH, HEIGHT - 2);
        // StdDraw.text(4, this.height - 1, "Round: " + this.round);
        // StdDraw.text(this.width - 7, this.height - 1, ENCOURAGEMENT[rand.nextInt(7)]);
        // StdDraw.text(this.width - 7, this.height - 1, ENCOURAGEMENT[rand.nextInt(7)]);

        StdDraw.show();
    }
    public void showSeed(String s) {
        StdDraw.clear(Color.BLACK);
        StdDraw.setPenColor(Color.GRAY);
        Font font = new Font("Monaco", Font.BOLD, 30);
        StdDraw.setFont(font);
        StdDraw.text(WIDTH / 2, HEIGHT / 2 + 10, "seed");
        StdDraw.text(WIDTH / 2, HEIGHT / 2, s);
        StdDraw.text(WIDTH / 2, HEIGHT / 2 - 3, "press s to enter");
        StdDraw.show();
    }

    public void playGame(BerkeleyGame bg, String data) {
        Character move = ' ';
        while (!bg.gameOver) {
            if (StdDraw.hasNextKeyTyped()) {
                move = StdDraw.nextKeyTyped();

                // if player initiates quit
                while (move.equals(':')) {
                    if (StdDraw.hasNextKeyTyped()) {
                        move = StdDraw.nextKeyTyped();
                        if (move.equals('Q') || move.equals('q')) {
                            saveGame(data);
                            System.exit(0);
                        }
                    }
                }
                data += move;
                bg.updateGame(move);
                ter.renderFrame(bg.berkeleyWorld.getCampus());
            }
        }
    }

    public void saveGame(String data) {
        try {
            File save = new File("C:\\Users\\15109\\Documents\\Programming\\CS61B\\fa22-s746\\proj3\\save.txt");
            FileWriter writer = new FileWriter(save);
            writer.write(data);
            writer.close();
        } catch (IOException e) {
            System.out.println("Unable to save at this time. Try again later.");
        }
    }

    public String loadFile() {
        String data = "";
        try {
            File save = new File("C:\\Users\\15109\\Documents\\Programming\\CS61B\\fa22-s746\\proj3\\save.txt");
            Scanner reader = new Scanner(save);
            while (reader.hasNextLine()) {
                data = reader.nextLine();
            }
            reader.close();
        } catch (FileNotFoundException e) {
                System.out.println("Unable to load at this time. Try again later.");
        }
        return data;
    }

    public void loadGame(String data) {
        int start = data.indexOf('n') + 1;
        int gameplay = data.indexOf('s') + 1;
        String seed = "";
        for (int i = start; i < data.indexOf('s'); i++) {
            seed += data.charAt(i);
        }
        ter.initialize(WIDTH, HEIGHT);
        BerkeleyGame bg = new BerkeleyGame(90, 45, Integer.parseInt(seed));
        bg.startGame();
        ter.renderFrame(bg.berkeleyWorld.getCampus());
        for (int i = gameplay; i < data.length(); i++) {
            bg.updateGame(data.charAt(i));
        }
        ter.renderFrame(bg.berkeleyWorld.getCampus());
        playGame(bg, data);
    }
    
    public TETile[][] interactWithInputString(String input) {
        /* interactWithInputString runs the engine using the input, and returns a 2D tile representation of the
    world that would have been drawn if the same inputs had been given to interactWithKeyboard().*/
        loadGame(input);

        TETile[][] finalWorldFrame = null;
        return finalWorldFrame;
    }
}
