(define (problem rearrangement-notyping) 
    (:domain rearrangement-notyping)

    (:objects
    
	pawn-0
	monkey-1
	robot
	loc-0-0
	loc-0-1
	loc-0-2
	loc-0-3
	loc-1-0
	loc-1-1
	loc-1-2
	loc-1-3
	loc-2-0
	loc-2-1
	loc-2-2
	loc-2-3
	loc-3-0
	loc-3-1
	loc-3-2
	loc-3-3
    )

    (:init
    
	(ispawn pawn-0)
	(ismonkey monkey-1)
	(isrobot robot)
	(at pawn-0 loc-2-3)
	(at monkey-1 loc-2-2)
	(at robot loc-0-3)
	(handsfree robot)

    ; action literals
    
	(pick pawn-0)
	(place pawn-0)
	(pick monkey-1)
	(place monkey-1)
	(moveto loc-0-0)
	(moveto loc-0-1)
	(moveto loc-0-2)
	(moveto loc-0-3)
	(moveto loc-1-0)
	(moveto loc-1-1)
	(moveto loc-1-2)
	(moveto loc-1-3)
	(moveto loc-2-0)
	(moveto loc-2-1)
	(moveto loc-2-2)
	(moveto loc-2-3)
	(moveto loc-3-0)
	(moveto loc-3-1)
	(moveto loc-3-2)
	(moveto loc-3-3)
    )

    (:goal (and  (holding pawn-0)  (at pawn-0 loc-0-1) ))
)
    