import random
import sys

import arcade
import numpy as np
import os
import time

# --- Constants ---
SPRITE_SCALING_PLAYER = 0.24

BOID_COUNT = 60

SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500
SCREEN_TITLE = "Reynolds Boids"

BOID_CONFIG={
    "radius":3,
    "base":4,
    "color": arcade.color.RED,
    "maxacceleration":10,
    "velocity":5,
    "maxvelocity":20,
    "perception":500,
    "dangerradius":30,
    "obstacleradius":20,
    "obstaclecolor": arcade.color.BABY_BLUE,
}
class Boid:
    position=np.zeros(2).astype(int)
    velocity=np.zeros(2).astype(int)
    acceleration=np.zeros(2)

    def __init__(self,game):
        self.game=game
        while True:
            self.position=np.array(
                        (random.randint(0+BOID_CONFIG["radius"],SCREEN_WIDTH-0+BOID_CONFIG["radius"]),
                                        random.randint(0+BOID_CONFIG["radius"],SCREEN_WIDTH-0+BOID_CONFIG["radius"]))
                         )
            if self.verifyPosition(self.position):
                break
        v=random.randint(1,BOID_CONFIG["maxvelocity"])
        orientation=random.random()*2*np.pi

        self.velocity=np.array( (
                        int(v*np.cos(orientation)),
                        int(v*np.sin(orientation))
        ) ) #norm=v, angle=orientation


    def x(self):
        return self.position[0]
    def y(self):
        return self.position[1]

    def velocity_val(self):
        return np.linalg.norm(self.velocity)

    def normalize_speed(self,speed):
        n=np.linalg.norm(speed)
        if n>BOID_CONFIG["maxvelocity"]:
            speed = (speed / n) * BOID_CONFIG["maxvelocity"]
        return np.around(speed).astype(int)

    def normalize_acceleration(self,acc):
        n=np.linalg.norm(acc)
        if n>BOID_CONFIG["maxacceleration"]:
            acc = (acc / n) * BOID_CONFIG["maxacceleration"]
        return acc

    def draw(self):

        rad=np.arctan(self.velocity[1]/(self.velocity[0]+sys.float_info.epsilon)) - \
                (1 - np.sign(self.velocity[0])) * np.sign(self.velocity[0]) * np.pi / 2


        # rad=np.arctan(self.y()/(self.x()+sys.float_info.epsilon))
        theta=np.arccos(1-BOID_CONFIG["base"]**2/(2*BOID_CONFIG["radius"]**2))

        x1,y1=(self.x()+BOID_CONFIG["radius"]*np.cos(rad),self.y()+BOID_CONFIG["radius"]*np.sin(rad))
        u1=rad-np.pi+theta/2
        u2=rad+np.pi-theta/2
        x2,y2=(self.x()+BOID_CONFIG["radius"]*np.cos(u1),self.y()+BOID_CONFIG["radius"]*np.sin(u1))
        x3,y3=(self.x()+BOID_CONFIG["radius"]*np.cos(u2),self.y()+BOID_CONFIG["radius"]*np.sin(u2))

        self.shape=arcade.create_triangles_filled_with_colors([(x1,y1),(x2,y2),(x3,y3)],[BOID_CONFIG["color"],BOID_CONFIG["color"],BOID_CONFIG["color"]])
        #arcade.draw_triangle_filled(x1,y1,x2+(x1-x2)*2/3,y2+(y1-y2)*2/3,x3+(x1-x3)*2/3,y3+(x1-y3)*2/3,BOID_CONFIG["color2"])

    def verifyPosition(self,position):
        for b in self.game.boids_list:
            if b==self:
                continue
            else:
                d=np.linalg.norm(b.position-position)
                if d<=BOID_CONFIG["radius"]:
                    return False
        return True
    def move(self):
        self.acceleration+=self.align()
        self.acceleration+=self.cohesion()
        self.acceleration+=self.separation()
        self.acceleration+=self.checkdanger()

        self.acceleration=self.normalize_acceleration(self.acceleration)

        self.velocity+=np.around(self.acceleration).astype(int)
        self.velocity=self.normalize_speed(self.velocity)
        self.position+=self.velocity

        if self.position[0]<0 or self.position[0]>SCREEN_WIDTH:
            self.position -= self.velocity
            self.velocity[0]=-self.velocity[0]
            self.velocity[1]=0
            self.position += self.velocity
        if self.position[1]<0 or self.position[1]>SCREEN_HEIGHT:
            self.position -= self.velocity
            self.velocity[1]=-self.velocity[1]
            self.velocity[0]=0
            self.position += self.velocity
        if self.in_obstacle(self.position):
            self.position -= self.velocity
            self.velocity=-self.velocity
            self.position += self.velocity


    def distance(self,position):
        return np.linalg.norm(self.position-position)

    def visible_boids(self):
        res=[]
        for b in self.game.boids_list:
            if b!=self and self.distance(b.position)<=BOID_CONFIG["perception"]:
                res.append(b)
        return res

    def align(self):
        steering=np.zeros(2)
        boids=self.visible_boids()
        avg=np.zeros(2)
        for b in boids:
            avg+=b.velocity
        if len(boids)>0:
            avg /= len(boids)
            steering=avg-self.velocity
            steering = self.normalize_acceleration(steering)
        return steering

    def cohesion(self):
        steering=np.zeros(2)
        boids=self.visible_boids()
        center=np.zeros(2)
        for b in boids:
            center+=b.position
        if len(boids)>0:
            center/=len(boids)
            vect_to_center=center-np.array(self.position)

            d=np.linalg.norm(vect_to_center)

            steering=vect_to_center-self.velocity
            steering = self.normalize_acceleration(steering)
        return steering

    def separation(self):
        steering=np.zeros(2)
        boids=self.visible_boids()
        avg=np.zeros(2)
        for b in boids:
            diff = self.position - b.position
            #diff = diff / int(self.distance(b.position))
            avg+=diff
        if len(boids)>0:
            avg/=len(boids)
            steering = avg - self.velocity
            steering = self.normalize_acceleration(steering)
        return steering

    def in_obstacle(self,position):
        for o in self.game.obstacles_list:
            if np.linalg.norm(o-position)<=BOID_CONFIG["obstacleradius"]:
                return True
        return False

    def checkdanger(self):
        steering=np.zeros(2)
        if np.linalg.norm(self.velocity)>0:
            v_norm=self.velocity / np.linalg.norm(self.velocity)
        else:
            v_norm = self.velocity
        checkpos=self.position+self.velocity+v_norm *BOID_CONFIG["perception"]
        checkpos_danger=self.position+self.velocity+v_norm *BOID_CONFIG["dangerradius"]

        if checkpos[0]<0 or checkpos[0]>SCREEN_WIDTH or (self.in_obstacle(checkpos) and self.velocity[0]>self.velocity[1]):
            steering+=np.array((-v_norm[1],v_norm[0]))*BOID_CONFIG["maxacceleration"]
        if checkpos[1] < 0 or checkpos[1] > SCREEN_HEIGHT:
            steering+=np.array((v_norm[1],-v_norm[0]))*BOID_CONFIG["maxacceleration"]

        if checkpos_danger[0]<0 or checkpos_danger[0]>SCREEN_WIDTH or (self.in_obstacle(checkpos) and self.velocity[0]<=self.velocity[1]):
            steering+=np.array((-v_norm[1],0))*BOID_CONFIG["maxacceleration"]
        if checkpos_danger[1] < 0 or checkpos_danger[1] > SCREEN_HEIGHT :
            steering+=np.array((0,-v_norm[0]))*BOID_CONFIG["maxacceleration"]


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
        self.obstacles_list=[]
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
        for ob in self.obstacles_list:
            arcade.draw_circle_filled(ob[0],ob[1],BOID_CONFIG["obstacleradius"],BOID_CONFIG["obstaclecolor"])

    def on_mouse_motion(self, x, y, dx, dy):
        """ Handle Mouse Motion """

        # Move the center of the player sprite to match the mouse x, y
        self.player_sprite.center_x = x
        self.player_sprite.center_y = y

    def on_mouse_press(self, x, y, button, modifiers):
        p=np.array((x,y))
        for o in self.obstacles_list:
            if np.linalg.norm(o-p)<=BOID_CONFIG["obstacleradius"]:
                self.obstacles_list.remove(o)
                return

        self.obstacles_list.append(p)

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