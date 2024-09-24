# Create a Destructible Object with Blast 🪨🪨🪨🪨⛰️⛰️🧗‍♀️
<!-- Docs/assets/blast/blast.gif -->
![Blast](assets/blast/blast.gif)

## Step 1: Install Blast 💥

Follow the steps to install Blast, a tool used for creating destructible objects in 3D environments. 

to install Blast, follow the steps below:
 * Open the Isaac Sim application.
    * Click on the `Window` tab in the top menu.
    * Select `extensions`.
    * Search `Blast`.
    * Click on the `install` button to install the Blast extension.
    * besure to click `Autoload` to enable the extension to load automatically when Isaac Sim starts.

![Blast Installation](assets/blast/2024-09-23-14-51-20.png)

## Step 2: Create a New Project

Start by creating a new project in Issac Sim. This project will serve as the environment for creating and testing destructible objects.

## Step 3: Create a Physics Plane

Add a physics plane to act as the ground for your scene. This plane will serve as a base for testing the destructible properties of objects.

## Step 4: Create a Destructible Object

Add a 3D object (e.g., a cube) that you want to make destructible. Adjust the size of the object as needed.
 
Be sure  that you add a mesh object other wise this will not work.

- **Scaling the Object:** Press the `R` key to enter the scale mode, then drag along the axis to adjust the object's size.

![Scaling Object](assets/blast/2024-09-23-14-54-38.png)

## Step 5: Assign Physics Components

Assign a Rigid body and Collider to the object to make it interact with physics.

- **Rigidbody:** Allows the object to be affected by gravity and forces.
- **Collider:** Defines the shape of the object for physical collisions.

You can test this by running the simulation; the object should fall if positioned above the ground plane.

![Physics Setup](assets/blast/2024-09-23-14-53-18.png)

## Step 6: Set Up the Object to Be Destructible

Configure the object to be destructible by adjusting its properties with the Blast settings.

![Destructible Setup](assets/blast/2024-09-23-14-49-59.png)

## Step 7: Test the Destruction

Upon setting the object to be destructible, it will return to its original scale visually. However, when interacted with (e.g., hit or collided with another object), it will break into smaller pieces as initially scaled.

![Destruction Preview](assets/blast/2024-09-23-15-01-39.png)



you can see a demo of the process on youtube [here](https://www.youtube.com/watch?v=your-video-id)

[![](assets/blast/Screenshot.png)](https://youtu.be/1ShDHfw6nUQ)