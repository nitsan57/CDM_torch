(define (problem rearrangement-notyping) 
    (:domain rearrangement-notyping)

    (:objects
    
	monkey-0
	monkey-1
	pawn-2
	robot
	loc-0-0
	loc-0-1
	loc-0-2
	loc-0-3
	loc-0-4
	loc-1-0
	loc-1-1
	loc-1-2
	loc-1-3
	loc-1-4
	loc-2-0
	loc-2-1
	loc-2-2
	loc-2-3
	loc-2-4
	loc-3-0
	loc-3-1
	loc-3-2
	loc-3-3
	loc-3-4
	loc-4-0
	loc-4-1
	loc-4-2
	loc-4-3
	loc-4-4
    )

    (:init
    
	(ismonkey monkey-0)
	(ismonkey monkey-1)
	(ispawn pawn-2)
	(isrobot robot)
	(at monkey-0 loc-3-0)
	(at monkey-1 loc-0-4)
	(at pawn-2 loc-3-3)
	(at robot loc-4-0)
	(handsfree robot)

    ; action literals
    
	(pick monkey-0)
	(place monkey-0)
	(pick monkey-1)
	(place monkey-1)
	(pick pawn-2)
	(place pawn-2)
	(moveto loc-0-0)
	(moveto loc-0-1)
	(moveto loc-0-2)
	(moveto loc-0-3)
	(moveto loc-0-4)
	(moveto loc-1-0)
	(moveto loc-1-1)
	(moveto loc-1-2)
	(moveto loc-1-3)
	(moveto loc-1-4)
	(moveto loc-2-0)
	(moveto loc-2-1)
	(moveto loc-2-2)
	(moveto loc-2-3)
	(moveto loc-2-4)
	(moveto loc-3-0)
	(moveto loc-3-1)
	(moveto loc-3-2)
	(moveto loc-3-3)
	(moveto loc-3-4)
	(moveto loc-4-0)
	(moveto loc-4-1)
	(moveto loc-4-2)
	(moveto loc-4-3)
	(moveto loc-4-4)
    )

    (:goal (and  (at monkey-1 loc-4-0)  (at pawn-2 loc-3-4) ))
)
    