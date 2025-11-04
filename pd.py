

def main():
    p1_points = 0
    p2_points = 0

    play = True

    print('\nWelcome to the Iterated Prisoner\'s Dilemma.')
    print('Each player has the choice to defect or cooperate.')
    print('If both players defect, you each gain little from the interaction (+1 each)')
    print('If 1 player defects and the other cooperates, the defecter will win big (defecter gains +5, cooperator does not gain)')
    print('If both players cooperate, they both gain modestly (+3 each)\n')
    
    while True:
            inp = (input('Begin Playing? (y/n): '))
            if inp.isalpha():
                if inp == 'n': play = False; break
                if inp == 'y': break


    while play:

        # Input actions
        p1_action = ''
        p2_action = ''
        while True:
            p1_action = (input('Player 1\'s action (d: Defect, c: Cooperate): '))
            if p1_action.isalpha() and (p1_action == 'd' or p1_action == 'c'): break
        while True:
            p2_action = (input('Player 2\'s action (d: Defect, c: Cooperate): '))
            if p2_action.isalpha() and (p2_action == 'd' or p2_action == 'c'): break
        print()

        # Calculate result
        if p1_action == p2_action == 'd':
            print('Both players defect: +1 for each player')
            p1_points += 1; p2_points += 1
        elif p1_action == p2_action == 'c':
            print('Both players cooperate: +3 for each player')
            p1_points += 3; p2_points += 3
        elif p1_action == 'd' and p2_action != 'd':
            print('Player 1 defects and player 2 cooperates: P1 +5, P2 +0')
            p1_points += 5
        elif p2_action == 'd' and p1_action != 'd':
            print('Player 2 defects and player 1 cooperates: P1 +0, P2 +5')
            p2_points += 5
        else:
            print('Unexpected answer. Repeat round')
            continue

        # Print results
        print(f'Player 1 has {p1_points} points currently.')
        print(f'Player 2 has {p2_points} points currently.')

        # Play again inputs
        while True:
            inp = (input('Play again? (y/n): '))
            if inp.isalpha():
                if inp == 'n': play = False; break
                if inp == 'y': break
                
    print('Game ended.')

if __name__ == '__main__':
    main()