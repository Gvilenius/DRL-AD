##############################################################################
#
#    file                 : Makefile
#    created              : 2019年 08月 15日 星期四 19:36:24 CST
#    copyright            : (C) 2002 Xinning Zhou
#
##############################################################################
#
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
#
##############################################################################

ROBOT       = ljf
MODULE      = ${ROBOT}.so
MODULEDIR   = drivers/${ROBOT}
SOURCES     = ${ROBOT}.cpp env.cpp

SHIPDIR     = drivers/${ROBOT}
SHIP        = ${ROBOT}.xml kc-gt40.rgb logo.rgb
SHIPSUBDIRS = 

PKGSUBDIRS  = ${SHIPSUBDIRS}
src-robots-lemon_PKGFILES = $(shell find * -maxdepth 0 -type f -print)
src-robots-lemon_PKGDIR   = ${PACKAGE}-${VERSION}/$(subst ${TORCS_BASE},,$(shell pwd))

include ${MAKE_DEFAULT}
