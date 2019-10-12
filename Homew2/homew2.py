import random
import arcade
import numpy as np
import os


# --- Constants ---
SPRITE_SCALING_PLAYER = 0.3

BOID_COUNT = 7

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Reynolds Boids"

BOID_CONFIG={
    "radius":10,
    "base":7,
    "color": arcade.color.RED,
    "color2": arcade.color.BLUE,
    "maxacceleration":20,
    "velocity":10,
    "maxvelocity":20,
    "perception":100,
}
class Boid:
    position=np.zeros(2)
    velocity=np.zeros(2)

    def __init__(self,game):
        self.game=game
        while True:
            self.x=random.randint(0+BOID_CONFIG["radius"],SCREEN_WIDTH-0+BOID_CONFIG["radius"])
            self.y=random.randint(0+BOID_CONFIG["radius"],SCREEN_WIDTH-0+BOID_CONFIG["radius"])
            if self.verifyPosition(self.x,self.y):
                break
        self.velocity=BOID_CONFIG["velocity"]
        self.acceleration=BOID_CONFIG["acceleration"]
        self.orientation=random.randint(0,360) #degrees
    def x(self):
        return self.position[0]
    def y(self):
        return self.position[1]
    def velocity_val(self):
        return np.linalg.norm(self.velocity)
    def draw(self):
        rad=self.orientation*(np.pi/180)
        theta=np.arccos(1-BOID_CONFIG["base"]**2/(2*BOID_CONFIG["radius"]**2))

        x1,y1=(self.x+BOID_CONFIG["radius"]*np.cos(rad),self.y+BOID_CONFIG["radius"]*np.sin(rad))
        u1=rad-np.pi+theta/2
        u2=rad+np.pi-theta/2
        x2,y2=(self.x+BOID_CONFIG["radius"]*np.cos(u1),self.y+BOID_CONFIG["radius"]*np.sin(u1))
        x3,y3=(self.x+BOID_CONFIG["radius"]*np.cos(u2),self.y+BOID_CONFIG["radius"]*np.sin(u2))

        self.shape=arcade.create_triangles_filled_with_colors([(x1,y1),(x2,y2),(x3,y3)],[BOID_CONFIG["color"],BOID_CONFIG["color"],BOID_CONFIG["color"]])
        #arcade.draw_triangle_filled(x1,y1,x2+(x1-x2)*2/3,y2+(y1-y2)*2/3,x3+(x1-x3)*2/3,y3+(x1-y3)*2/3,BOID_CONFIG["color2"])

    def verifyPosition(self,x,y):
        for b in self.game.boids_list:
            if b==self:
                continue
            else:
                d=np.sqrt((b.x-x)**2+(b.y-y)**2)
                if d<=BOID_CONFIG["radius"]:
                    return False
        return True
    def move(self):
        rad = self.orientation * (np.pi / 180)
        newx,newy=self.x+BOID_CONFIG["velocity"]*np.cos(rad),self.y+BOID_CONFIG["velocity"]*np.sin(rad)

        self.x,self.y=newx,newy
    def distance(self,x,y):
        return np.linalg.norm((self.x,self.y)-(x,y))

    def visible_boids(self):
        res=[]
        for b in self.game.boids_list:
            if b!=self and self.distance(b.x,b.y)<=BOID_CONFIG["perception"]:
                res.append(b)
        return res
    def velocity_vector(self):
        rad = self.orientation * (np.pi / 180)
        res=np.ndarray(self.velocity*np.cos(rad),self.velocity*np.sin(rad))
        return res
    def align(self):
        steering=np.zeros(2)
        boids=self.visible_boids()
        avg=np.zeros(2)
        for b in boids:
            avg+=b.velocity_vector()
        if len(boids)>0:
            avg /= len(boids)
            avg = (avg/np.linalg.norm(avg))*BOID_CONFIG["maxvelocity"]
            steering=avg-self.velocity_vector()
        return steering
    def cohesion(self):
        steering=np.zeros(2)
        boids=self.visible_boids()
        center=np.zeros(2)
        for b in boids:
            center+=np.ndarray(self.x,self.y)
        if len(boids)>0:
            center/=len(boids)
            vect_to_center=center-np.ndarray((self.x,self.y))
            d=np.linalg.norm(vect_to_center)
            if d>0:
                vect_to_center=(vect_to_center/d)*self.velocity
            steering=vect_to_center-self.velocity_vector()
            if np.linalg.norm(steering)>self.acceleration:
                steering=(steering/np.linalg.norm(steering))*self.acceleration
            return steering


class MyGame(arcade.Window):
    """ Our custom Window Class"""

    def __init__(self):
        """ Initializer """
        # Call the parent class initializer
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)

        # Set the working directory (where we expect to find files) to the same
        # directory this .py file is in. You can leave this out of your own
        # code, but it is needed to easily run the examples using "python -m"
        # as mentioned at the top of this program.
        file_path = os.path.dirname(os.path.abspath(__file__))
        os.chdir(file_path)

        # Variables that will hold sprite lists
        self.player_list = None
        self.coin_list = None

        # Set up the player info
        self.player_sprite = None
        self.score = 0

        # Don't show the mouse cursor
        self.set_mouse_visible(False)

        arcade.set_background_color(arcade.color.AMAZON)

    def setup(self):
        """ Set up the game and initialize the variables. """

        # Sprite lists
        self.player_list = arcade.SpriteList()
        self.shape_list = arcade.ShapeElementList()
        self.boids_list=[]
        # Score
        self.score = 0

        # Set up the player
        # Character image from kenney.nl
        self.player_sprite = arcade.Sprite("images/playerShip2_orange.png", SPRITE_SCALING_PLAYER)
        self.player_sprite.center_x = 50
        self.player_sprite.center_y = 50
        self.player_list.append(self.player_sprite)

        # Create the boids
        for i in range(BOID_COUNT):
            bd = Boid(self)
            bd.draw()
            self.boids_list.append(bd)
            self.shape_list.append(bd.shape)

    def on_draw(self):
        """ Draw everything """
        arcade.start_render()
        self.player_list.draw()

        # Put the text on the screen.
        output = f"Boids: {len(self.boids_list)}"
        arcade.draw_text(output, 10, 20, arcade.color.WHITE, 14)
        self.shape_list.draw()
    def on_mouse_motion(self, x, y, dx, dy):
        """ Handle Mouse Motion """

        # Move the center of the player sprite to match the mouse x, y
        self.player_sprite.center_x = x
        self.player_sprite.center_y = y

    def update(self, delta_time):
        """ Movement and game logic """

        # Call update on all sprites (The sprites don't do much in this
        # example though.)
        # self.coin_list.update()

        # # Generate a list of all sprites that collided with the player.
        # coins_hit_list = arcade.check_for_collision_with_list(self.player_sprite, self.coin_list)
        #
        # # Loop through each colliding sprite, remove it, and add to the score.
        # for coin in coins_hit_list:
        #     coin.kill()
        #     self.score += 1
        self.shape_list = arcade.ShapeElementList()

        for b in self.boids_list:
            b.move()
            b.draw()
            self.shape_list.append(b.shape)


def main():
    """ Main method """
    window = MyGame()
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()