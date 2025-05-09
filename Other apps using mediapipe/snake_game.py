import pygame
import time
from cvzone.HandTrackingModule import HandDetector
import cv2
import random

# CV2
cap = cv2.VideoCapture(0)
# Hand Detector
detectorHand = HandDetector(detectionCon=0.8, maxHands=1)

pygame.init()

# Define colors
white = (255, 255, 255)
yellow = (255, 255, 102)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 255, 0)
blue = (50, 153, 213)

# Set display width and height
dis_width = 800
dis_height = 600

dis = pygame.display.set_mode((dis_width, dis_height))
pygame.display.set_caption('Snake Game')

clock = pygame.time.Clock()

snake_block = 10
snake_speed = 20

font_style = pygame.font.SysFont(None, 50)

# Function to display the snake
def our_snake(snake_block, snake_list):
    for x in snake_list:
        pygame.draw.rect(dis, black, [x[0], x[1], snake_block, snake_block])

# Function to display the message
def message(msg, color, score=None):
    mesg = font_style.render(msg, True, color)
    dis.blit(mesg, [dis_width / 6, dis_height / 3])
    if score is not None:
        score_text = font_style.render("Score: " + str(score), True, color)
        dis.blit(score_text, [dis_width / 6, dis_height / 3 + 50])




# Function to display the score
def show_score(score):
    score_text = font_style.render("Score: " + str(score), True, black)
    dis.blit(score_text, [0, 0])

# Main function
def gameLoop():
    game_over = False

    x1 = dis_width / 2
    y1 = dis_height / 2

    x1_change = 0
    y1_change = 0

    snake_List = []
    Length_of_snake = 1

    foodx = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
    foody = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0

    score = 0

    while not game_over:
        # Get image frame
        success, img = cap.read()
        img = cv2.flip(img, 1)
        hands, img = detectorHand.findHands(img)
        if hands:
            hand = hands[0]
            fingers = detectorHand.fingersUp(hand)

            # Control the snake with hand gestures
            if fingers == [1, 0, 0, 0, 0]:  # Move left
                x1_change = -snake_block
                y1_change = 0
            elif fingers == [0, 0, 0, 0, 1]:  # Move right
                x1_change = snake_block
                y1_change = 0
            elif fingers == [0, 1, 0, 0, 0]:  # Move up
                y1_change = -snake_block
                x1_change = 0
            elif fingers == [0, 0, 0, 0, 0]:  # Move down
                y1_change = snake_block
                x1_change = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:  # Quit the game
                    pygame.quit()
                    quit()
                elif event.key == pygame.K_r:  # Restart the game
                    gameLoop()

        x1 += x1_change
        y1 += y1_change
        dis.fill(blue)
        pygame.draw.rect(dis, green, [foodx, foody, snake_block, snake_block])
        snake_Head = []
        snake_Head.append(x1)
        snake_Head.append(y1)
        snake_List.append(snake_Head)
        if len(snake_List) > Length_of_snake:
            del snake_List[0]

        # for x in snake_List[:-1]:
        #     if x == snake_Head:
        #         game_over = True

        our_snake(snake_block, snake_List)
        show_score(score)

        pygame.display.update()

        if x1 == foodx and y1 == foody:
            foodx = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
            foody = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0
            Length_of_snake += 2
            score += 10

        # Game over conditions
        if x1 >= dis_width or x1 < 0 or y1 >= dis_height or y1 < 0:
            game_over = True

        clock.tick(snake_speed)

    while game_over:

        dis.fill(blue)
        message("You Lost! Press Q-Quit or R-Play Again", red, score)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:  # Quit the game
                    pygame.quit()
                    quit()
                elif event.key == pygame.K_r:  # Restart the game
                    game_over = False
                    gameLoop()

gameLoop()
