(define (problem rearrangement-notyping) 
    (:domain rearrangement-notyping)

    (:objects
    
	pawn-0
	pawn-1
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
	loc-3-0
	loc-3-1
	loc-3-2
	loc-3-3
	loc-4-0
	loc-4-1
	loc-4-2
	loc-4-3
    )

    (:init
    
	(ispawn pawn-0)
	(ispawn pawn-1)
	(isbear bear-2)
	(ispawn pawn-3)
	(isrobot robot)
	(at pawn-0 loc-4-2)
	(at pawn-1 loc-2-1)
	(at bear-2 loc-4-2)
	(at pawn-3 loc-1-0)
	(at robot loc-3-1)
	(handsfree robot)

    ; action literals
    
	(pick pawn-0)
	(place pawn-0)
	(pick pawn-1)
	(place pawn-1)
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
	(moveto loc-3-0)
	(moveto loc-3-1)
	(moveto loc-3-2)
	(moveto loc-3-3)
	(moveto loc-4-0)
	(moveto loc-4-1)
	(moveto loc-4-2)
	(moveto loc-4-3)
    )

    (:goal (and  (at bear-2 loc-3-0) ))
)
    