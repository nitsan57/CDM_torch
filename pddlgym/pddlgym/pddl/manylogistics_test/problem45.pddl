


(define (problem logistics-c24-s3-p23-a54)
(:domain logistics-strips)
(:objects a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16 a17 a18 a19 a20 a21 a22 a23 a24 a25 a26 a27 a28 a29 a30 a31 a32 a33 a34 a35 a36 a37 a38 a39 a40 a41 a42 a43 a44 a45 a46 a47 a48 a49 a50 a51 a52 a53 
          c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16 c17 c18 c19 c20 c21 c22 c23 
          t0 t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 t11 t12 t13 t14 t15 t16 t17 t18 t19 t20 t21 t22 t23 
          l00 l01 l02 l10 l11 l12 l20 l21 l22 l30 l31 l32 l40 l41 l42 l50 l51 l52 l60 l61 l62 l70 l71 l72 l80 l81 l82 l90 l91 l92 l100 l101 l102 l110 l111 l112 l120 l121 l122 l130 l131 l132 l140 l141 l142 l150 l151 l152 l160 l161 l162 l170 l171 l172 l180 l181 l182 l190 l191 l192 l200 l201 l202 l210 l211 l212 l220 l221 l222 l230 l231 l232 
          p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 
)
(:init
(AIRPLANE a0)
(AIRPLANE a1)
(AIRPLANE a2)
(AIRPLANE a3)
(AIRPLANE a4)
(AIRPLANE a5)
(AIRPLANE a6)
(AIRPLANE a7)
(AIRPLANE a8)
(AIRPLANE a9)
(AIRPLANE a10)
(AIRPLANE a11)
(AIRPLANE a12)
(AIRPLANE a13)
(AIRPLANE a14)
(AIRPLANE a15)
(AIRPLANE a16)
(AIRPLANE a17)
(AIRPLANE a18)
(AIRPLANE a19)
(AIRPLANE a20)
(AIRPLANE a21)
(AIRPLANE a22)
(AIRPLANE a23)
(AIRPLANE a24)
(AIRPLANE a25)
(AIRPLANE a26)
(AIRPLANE a27)
(AIRPLANE a28)
(AIRPLANE a29)
(AIRPLANE a30)
(AIRPLANE a31)
(AIRPLANE a32)
(AIRPLANE a33)
(AIRPLANE a34)
(AIRPLANE a35)
(AIRPLANE a36)
(AIRPLANE a37)
(AIRPLANE a38)
(AIRPLANE a39)
(AIRPLANE a40)
(AIRPLANE a41)
(AIRPLANE a42)
(AIRPLANE a43)
(AIRPLANE a44)
(AIRPLANE a45)
(AIRPLANE a46)
(AIRPLANE a47)
(AIRPLANE a48)
(AIRPLANE a49)
(AIRPLANE a50)
(AIRPLANE a51)
(AIRPLANE a52)
(AIRPLANE a53)
(CITY c0)
(CITY c1)
(CITY c2)
(CITY c3)
(CITY c4)
(CITY c5)
(CITY c6)
(CITY c7)
(CITY c8)
(CITY c9)
(CITY c10)
(CITY c11)
(CITY c12)
(CITY c13)
(CITY c14)
(CITY c15)
(CITY c16)
(CITY c17)
(CITY c18)
(CITY c19)
(CITY c20)
(CITY c21)
(CITY c22)
(CITY c23)
(TRUCK t0)
(TRUCK t1)
(TRUCK t2)
(TRUCK t3)
(TRUCK t4)
(TRUCK t5)
(TRUCK t6)
(TRUCK t7)
(TRUCK t8)
(TRUCK t9)
(TRUCK t10)
(TRUCK t11)
(TRUCK t12)
(TRUCK t13)
(TRUCK t14)
(TRUCK t15)
(TRUCK t16)
(TRUCK t17)
(TRUCK t18)
(TRUCK t19)
(TRUCK t20)
(TRUCK t21)
(TRUCK t22)
(TRUCK t23)
(LOCATION l00)
(in-city  l00 c0)
(LOCATION l01)
(in-city  l01 c0)
(LOCATION l02)
(in-city  l02 c0)
(LOCATION l10)
(in-city  l10 c1)
(LOCATION l11)
(in-city  l11 c1)
(LOCATION l12)
(in-city  l12 c1)
(LOCATION l20)
(in-city  l20 c2)
(LOCATION l21)
(in-city  l21 c2)
(LOCATION l22)
(in-city  l22 c2)
(LOCATION l30)
(in-city  l30 c3)
(LOCATION l31)
(in-city  l31 c3)
(LOCATION l32)
(in-city  l32 c3)
(LOCATION l40)
(in-city  l40 c4)
(LOCATION l41)
(in-city  l41 c4)
(LOCATION l42)
(in-city  l42 c4)
(LOCATION l50)
(in-city  l50 c5)
(LOCATION l51)
(in-city  l51 c5)
(LOCATION l52)
(in-city  l52 c5)
(LOCATION l60)
(in-city  l60 c6)
(LOCATION l61)
(in-city  l61 c6)
(LOCATION l62)
(in-city  l62 c6)
(LOCATION l70)
(in-city  l70 c7)
(LOCATION l71)
(in-city  l71 c7)
(LOCATION l72)
(in-city  l72 c7)
(LOCATION l80)
(in-city  l80 c8)
(LOCATION l81)
(in-city  l81 c8)
(LOCATION l82)
(in-city  l82 c8)
(LOCATION l90)
(in-city  l90 c9)
(LOCATION l91)
(in-city  l91 c9)
(LOCATION l92)
(in-city  l92 c9)
(LOCATION l100)
(in-city  l100 c10)
(LOCATION l101)
(in-city  l101 c10)
(LOCATION l102)
(in-city  l102 c10)
(LOCATION l110)
(in-city  l110 c11)
(LOCATION l111)
(in-city  l111 c11)
(LOCATION l112)
(in-city  l112 c11)
(LOCATION l120)
(in-city  l120 c12)
(LOCATION l121)
(in-city  l121 c12)
(LOCATION l122)
(in-city  l122 c12)
(LOCATION l130)
(in-city  l130 c13)
(LOCATION l131)
(in-city  l131 c13)
(LOCATION l132)
(in-city  l132 c13)
(LOCATION l140)
(in-city  l140 c14)
(LOCATION l141)
(in-city  l141 c14)
(LOCATION l142)
(in-city  l142 c14)
(LOCATION l150)
(in-city  l150 c15)
(LOCATION l151)
(in-city  l151 c15)
(LOCATION l152)
(in-city  l152 c15)
(LOCATION l160)
(in-city  l160 c16)
(LOCATION l161)
(in-city  l161 c16)
(LOCATION l162)
(in-city  l162 c16)
(LOCATION l170)
(in-city  l170 c17)
(LOCATION l171)
(in-city  l171 c17)
(LOCATION l172)
(in-city  l172 c17)
(LOCATION l180)
(in-city  l180 c18)
(LOCATION l181)
(in-city  l181 c18)
(LOCATION l182)
(in-city  l182 c18)
(LOCATION l190)
(in-city  l190 c19)
(LOCATION l191)
(in-city  l191 c19)
(LOCATION l192)
(in-city  l192 c19)
(LOCATION l200)
(in-city  l200 c20)
(LOCATION l201)
(in-city  l201 c20)
(LOCATION l202)
(in-city  l202 c20)
(LOCATION l210)
(in-city  l210 c21)
(LOCATION l211)
(in-city  l211 c21)
(LOCATION l212)
(in-city  l212 c21)
(LOCATION l220)
(in-city  l220 c22)
(LOCATION l221)
(in-city  l221 c22)
(LOCATION l222)
(in-city  l222 c22)
(LOCATION l230)
(in-city  l230 c23)
(LOCATION l231)
(in-city  l231 c23)
(LOCATION l232)
(in-city  l232 c23)
(AIRPORT l00)
(AIRPORT l10)
(AIRPORT l20)
(AIRPORT l30)
(AIRPORT l40)
(AIRPORT l50)
(AIRPORT l60)
(AIRPORT l70)
(AIRPORT l80)
(AIRPORT l90)
(AIRPORT l100)
(AIRPORT l110)
(AIRPORT l120)
(AIRPORT l130)
(AIRPORT l140)
(AIRPORT l150)
(AIRPORT l160)
(AIRPORT l170)
(AIRPORT l180)
(AIRPORT l190)
(AIRPORT l200)
(AIRPORT l210)
(AIRPORT l220)
(AIRPORT l230)
(OBJ p0)
(OBJ p1)
(OBJ p2)
(OBJ p3)
(OBJ p4)
(OBJ p5)
(OBJ p6)
(OBJ p7)
(OBJ p8)
(OBJ p9)
(OBJ p10)
(OBJ p11)
(OBJ p12)
(OBJ p13)
(OBJ p14)
(OBJ p15)
(OBJ p16)
(OBJ p17)
(OBJ p18)
(OBJ p19)
(OBJ p20)
(OBJ p21)
(OBJ p22)
(at t0 l02)
(at t1 l11)
(at t2 l21)
(at t3 l30)
(at t4 l41)
(at t5 l50)
(at t6 l60)
(at t7 l72)
(at t8 l81)
(at t9 l90)
(at t10 l101)
(at t11 l111)
(at t12 l121)
(at t13 l131)
(at t14 l142)
(at t15 l150)
(at t16 l160)
(at t17 l170)
(at t18 l182)
(at t19 l190)
(at t20 l200)
(at t21 l212)
(at t22 l222)
(at t23 l232)
(at p0 l142)
(at p1 l42)
(at p2 l202)
(at p3 l51)
(at p4 l81)
(at p5 l70)
(at p6 l102)
(at p7 l90)
(at p8 l171)
(at p9 l52)
(at p10 l152)
(at p11 l51)
(at p12 l142)
(at p13 l232)
(at p14 l80)
(at p15 l142)
(at p16 l111)
(at p17 l231)
(at p18 l131)
(at p19 l80)
(at p20 l121)
(at p21 l12)
(at p22 l131)
(at a0 l20)
(at a1 l200)
(at a2 l140)
(at a3 l180)
(at a4 l170)
(at a5 l90)
(at a6 l230)
(at a7 l110)
(at a8 l40)
(at a9 l170)
(at a10 l180)
(at a11 l230)
(at a12 l50)
(at a13 l150)
(at a14 l70)
(at a15 l40)
(at a16 l30)
(at a17 l60)
(at a18 l200)
(at a19 l190)
(at a20 l140)
(at a21 l140)
(at a22 l40)
(at a23 l50)
(at a24 l210)
(at a25 l120)
(at a26 l110)
(at a27 l150)
(at a28 l100)
(at a29 l10)
(at a30 l180)
(at a31 l120)
(at a32 l210)
(at a33 l80)
(at a34 l60)
(at a35 l70)
(at a36 l90)
(at a37 l210)
(at a38 l180)
(at a39 l50)
(at a40 l70)
(at a41 l120)
(at a42 l50)
(at a43 l120)
(at a44 l30)
(at a45 l120)
(at a46 l80)
(at a47 l60)
(at a48 l180)
(at a49 l50)
(at a50 l10)
(at a51 l80)
(at a52 l110)
(at a53 l210)
)
(:goal
(and
(at p0 l151)
(at p1 l180)
(at p2 l90)
(at p3 l171)
(at p4 l202)
(at p5 l152)
(at p6 l220)
(at p7 l81)
(at p8 l62)
(at p9 l152)
(at p10 l192)
(at p11 l20)
(at p12 l182)
(at p13 l32)
(at p14 l212)
(at p15 l60)
(at p16 l61)
(at p17 l02)
(at p18 l180)
(at p19 l152)
(at p20 l80)
(at p21 l21)
(at p22 l222)
)
)
)


