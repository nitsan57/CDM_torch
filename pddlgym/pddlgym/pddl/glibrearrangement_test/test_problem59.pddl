(define (problem glibrearrangement) 
    (:domain glibrearrangement)

    (:objects
    
	monkey-0 - moveable
	pawn-1 - moveable
	monkey-2 - moveable
	bear-3 - moveable
	robot - moveable
	loc-0-0 - static
	loc-0-1 - static
	loc-0-2 - static
	loc-1-0 - static
	loc-1-1 - static
	loc-1-2 - static
	loc-2-0 - static
	loc-2-1 - static
	loc-2-2 - static
	loc-3-0 - static
	loc-3-1 - static
	loc-3-2 - static
    )

    (:init
    
	(IsMonkey monkey-0)
	(IsPawn pawn-1)
	(IsMonkey monkey-2)
	(IsBear bear-3)
	(IsRobot robot)
	(At monkey-0 loc-2-0)
	(At pawn-1 loc-0-1)
	(At monkey-2 loc-0-2)
	(At bear-3 loc-2-0)
	(At robot loc-0-2)
	(Handsfree robot)

    ; Action literals
    
	(Pick monkey-0)
	(Place monkey-0)
	(Pick pawn-1)
	(Place pawn-1)
	(Pick monkey-2)
	(Place monkey-2)
	(Pick bear-3)
	(Place bear-3)
	(MoveTo loc-0-0)
	(MoveTo loc-0-1)
	(MoveTo loc-0-2)
	(MoveTo loc-1-0)
	(MoveTo loc-1-1)
	(MoveTo loc-1-2)
	(MoveTo loc-2-0)
	(MoveTo loc-2-1)
	(MoveTo loc-2-2)
	(MoveTo loc-3-0)
	(MoveTo loc-3-1)
	(MoveTo loc-3-2)
    )

    (:goal (and  (At bear-3 loc-3-2)  (At monkey-2 loc-3-1)  (At pawn-1 loc-2-1) ))
)
    