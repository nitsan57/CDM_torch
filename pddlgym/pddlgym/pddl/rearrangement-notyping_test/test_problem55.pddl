(define (problem rearrangement-notyping) 
    (:domain rearrangement-notyping)

    (:objects
    
	monkey-0
	monkey-1
	bear-2
	pawn-3
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
    )

    (:init
    
	(ismonkey monkey-0)
	(ismonkey monkey-1)
	(isbear bear-2)
	(ispawn pawn-3)
	(isrobot robot)
	(at monkey-0 loc-0-1)
	(at monkey-1 loc-1-2)
	(at bear-2 loc-1-0)
	(at pawn-3 loc-2-0)
	(at robot loc-0-0)
	(handsfree robot)

    ; action literals
    
	(pick monkey-0)
	(place monkey-0)
	(pick monkey-1)
	(place monkey-1)
	(pick bear-2)
	(place bear-2)
	(pick pawn-3)
	(place pawn-3)
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
    )

    (:goal (and  (holding monkey-1) ))
)
    