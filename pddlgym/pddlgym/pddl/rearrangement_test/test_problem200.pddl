(define (problem rearrangement) 
    (:domain rearrangement)

    (:objects
    
	pawn-0 - moveable
	bear-1 - moveable
	robot - moveable
	loc-0-0 - static
	loc-0-1 - static
	loc-0-2 - static
	loc-0-3 - static
	loc-1-0 - static
	loc-1-1 - static
	loc-1-2 - static
	loc-1-3 - static
	loc-2-0 - static
	loc-2-1 - static
	loc-2-2 - static
	loc-2-3 - static
	loc-3-0 - static
	loc-3-1 - static
	loc-3-2 - static
	loc-3-3 - static
    )

    (:init
    
	(ispawn pawn-0)
	(isbear bear-1)
	(isrobot robot)
	(at pawn-0 loc-3-2)
	(at bear-1 loc-1-3)
	(at robot loc-2-3)
	(handsfree robot)

    ; action literals
    
	(pick pawn-0)
	(place pawn-0)
	(pick bear-1)
	(place bear-1)
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

    (:goal (and  (at pawn-0 loc-0-2) ))
)
    