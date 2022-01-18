(define (domain glibrearrangement)
  (:requirements :strips :typing)
  (:types moveable static)
  (:predicates
     (isrobot ?robot - moveable)
     (ispawn ?pawn - moveable)
     (isbear ?bear - moveable)
     (isgoal ?goal - static)
     (ismonkey ?monkey - moveable)
     (at ?obj - moveable ?loc - static)
     (holding ?obj - moveable)
     (handsfree ?robot - moveable)
     (moveto ?loc - static)
     (pick ?obj - moveable)
     (place ?obj - moveable)
  )

  ; (:actions moveto pick place)

  (:action movetonotholding
    :parameters (?robot - moveable ?start - static ?end - static)
    :precondition (and (moveto ?end)
                       (isrobot ?robot)
                       (handsfree ?robot)
                       (at ?robot ?start)
                  )
    :effect (and (not (at ?robot ?start))
                 (at ?robot ?end)
            )
  )

  (:action movetoholding
    :parameters (?robot - moveable ?obj - moveable ?start - static ?end - static)
    :precondition (and (moveto ?end)
                       (isrobot ?robot)
                       (holding ?obj)
                       (at ?robot ?start)
                       (at ?obj ?start)
                  )
    :effect (and (not (at ?robot ?start))
                 (at ?robot ?end)
                 (not (at ?obj ?start))
                 (at ?obj ?end)
            )
  )

  (:action pick
    :parameters (?robot - moveable ?obj - moveable ?loc - static)
    :precondition (and (pick ?obj)
                       (isrobot ?robot)
                       (handsfree ?robot)
                       (at ?robot ?loc)
                       (at ?obj ?loc)
                       (not (isrobot ?obj))
                  )
    :effect (and (holding ?obj)
                 (not (handsfree ?robot))
            )
  )

  (:action place
    :parameters (?robot - moveable ?obj - moveable)
    :precondition (and (place ?obj)
                       (isrobot ?robot)
                       (holding ?obj)
                  )
    :effect (and (not (holding ?obj))
                 (handsfree ?robot)
            )
  )
)