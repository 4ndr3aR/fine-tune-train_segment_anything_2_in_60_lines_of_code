#!/usr/bin/env python3

import sys
#import time
from datetime import datetime
from enum import Enum
#from colorama import init, Fore, Style  # Requires colorama: pip install colorama
from xtermcolor import colorize

#init()  # Initialize colorama for Windows compatibility

class LogLevel(Enum):
	TRACE	= 1
	VERBOSE	= 2
	DEBUG	= 3
	INFO	= 4
	WARNING	= 5
	ERROR	= 6
	FATAL	= 7

# Loose global variables for loglevels
trace	= LogLevel.TRACE
verbose	= LogLevel.VERBOSE
debug	= LogLevel.DEBUG
info	= LogLevel.INFO
warning	= LogLevel.WARNING
error	= LogLevel.ERROR
fatal	= LogLevel.FATAL

class Subsystem(Enum):
	SHAREDMEM	= 1
	THREADING	= 2
	QUEUES		= 3
	NETWORK		= 4
	TRAIN		= 5
	VALIDATE	= 6
	TEST		= 7
	DATALOADER	= 8
	PREDICT		= 9
	MAIN		= 10
	WIDEST____	= 256			# match this one to the widest (longest string) one, so that it can always be used to calculate padding... (don't prepend with _)

# Loose global variables for subsystems (for convenient keyword access)
threading	= Subsystem.THREADING
sharedmemory	= Subsystem.SHAREDMEM
queues		= Subsystem.QUEUES
network		= Subsystem.NETWORK
train		= Subsystem.TRAIN
validate	= Subsystem.VALIDATE
test		= Subsystem.TEST
dataloader	= Subsystem.DATALOADER
predict		= Subsystem.PREDICT
main		= Subsystem.MAIN

class Color(Enum):
	DARK_GRAY	= 240
	DIM_WHITE	= 245
	LIGHT_GREEN	= 10
	WHITE		= 15
	YELLOW		= 11
	RED		= 9
	MAGENTA		= 13
	LIGHT_BLUE	= 12
	LIGHT_CYAN	= 14
	LIGHT_YELLOW	= 227
	LIGHT_MAGENTA	= 207
	LIGHT_RED	= 203
	BRIGHT_WHITE	= 15	# Already in your list as WHITE, we'll keep it as an alias
	SKY_BLUE	= 39	# Example of a nice color
	LAVENDER	= 183	# And another
	PINK		= 218	# Pastel pink
	BABY_BLUE	= 159	# Pastel blue
	MINT		= 156	# Pastel green
	ORANGE		= 208	# Orange

# Color dictionary for log levels, now using the Color enum
loglevel_colors = { 
	LogLevel.TRACE:		Color.DARK_GRAY,
	LogLevel.VERBOSE:	Color.DIM_WHITE,
	LogLevel.DEBUG:		Color.LIGHT_GREEN,
	LogLevel.INFO:		Color.WHITE,
	LogLevel.WARNING:	Color.YELLOW,
	LogLevel.ERROR:		Color.RED,
	LogLevel.FATAL:		Color.MAGENTA,
}

# Color dictionary for subsystems, now using the Color enum
subsystem_colors = { 
	Subsystem.SHAREDMEM:	Color.LIGHT_BLUE,
	Subsystem.THREADING:	Color.LIGHT_CYAN,
	Subsystem.QUEUES:	Color.LIGHT_YELLOW,
	Subsystem.NETWORK:	Color.LIGHT_MAGENTA,
	Subsystem.TRAIN:	Color.LIGHT_RED,
	Subsystem.VALIDATE:	Color.ORANGE,
	Subsystem.TEST:		Color.SKY_BLUE,
	Subsystem.DATALOADER:	Color.PINK,
	Subsystem.PREDICT:	Color.BABY_BLUE,
	Subsystem.MAIN:		Color.MINT,
}


## Color dictionary for log levels
#loglevel_colors = { 
#    LogLevel.TRACE:     240,  # Dark gray
#    LogLevel.VERBOSE:   245,  # Dim white
#    LogLevel.DEBUG:     10,   # Light green
#    LogLevel.INFO:      15,   # White
#    LogLevel.WARNING:   11,   # Yellow
#    LogLevel.ERROR:     9,    # Red
#    LogLevel.FATAL:     13,   # Magenta
#}
#
## Color dictionary for subsystems
#subsystem_colors = { 
#    Subsystem.SHAREDMEM:        12,   # Light blue
#    Subsystem.THREADING:        14,   # Light cyan
#    Subsystem.QUEUES:           227,  # Light yellow
#    Subsystem.NETWORK:          207,  # Light magenta
#    Subsystem.TRAIN:            203,  # Light red
#    Subsystem.VALIDATE:         203,  # Light red
#}


## Xterm color codes for log levels
#loglevel_colors = { 
#    LogLevel.TRACE:     8,       # Light Black
#    LogLevel.VERBOSE:   7,       # White with dim style (xterm doesn't have a direct dim style, so we use white)
#    LogLevel.DEBUG:     10,      # Light Green
#    LogLevel.INFO:      15,      # White
#    LogLevel.WARNING:   11,      # Yellow
#    LogLevel.ERROR:     9,       # Red
#    LogLevel.FATAL:     13,      # Magenta
#}
#
## Xterm color codes for subsystems (customize as needed)
#subsystem_colors = { 
#    Subsystem.SHAREDMEM:        14,  # Light Blue
#    Subsystem.THREADING:        12,  # Light Cyan
#    Subsystem.QUEUES:           11,  # Light Yellow
#    Subsystem.NETWORK:          13,  # Light Magenta
#    Subsystem.TRAIN:            9,   # Light Red
#    Subsystem.VALIDATE:         9,   # Light Red
#}


# Enabled subsystems and their minimum log levels
enabled_subsystems = {
	Subsystem.THREADING:	LogLevel.DEBUG,
	Subsystem.QUEUES:	LogLevel.WARNING,
	Subsystem.NETWORK:	LogLevel.ERROR,
	Subsystem.SHAREDMEM:	LogLevel.TRACE,
	Subsystem.TRAIN:	LogLevel.INFO,
	Subsystem.VALIDATE:	LogLevel.INFO,
	Subsystem.TEST:		LogLevel.TRACE,
	Subsystem.DATALOADER:	LogLevel.WARNING,
	Subsystem.PREDICT:	LogLevel.INFO,
	Subsystem.MAIN:		LogLevel.TRACE,
	Subsystem.WIDEST____:	LogLevel.FATAL,		# match this one to the widest (longest string) one, so that it can always be used to calculate padding... (don't prepend with _)
	# Add more subsystems and their minimum log levels here
}

def dbgprint(subsystem, loglevel, *args, sep=' ', end='\n', flush=False):
    """
    Enhanced debug print function with color-coded output, subsystem filtering, and flexible formatting.
    """

    # Ensure the subsystem and loglevel are valid
    if not isinstance(subsystem, Subsystem):
        raise ValueError(f"Invalid subsystem: {subsystem}")
    if not isinstance(loglevel, LogLevel):
        raise ValueError(f"Invalid loglevel: {loglevel}")

    if subsystem not in enabled_subsystems or loglevel.value < enabled_subsystems[subsystem].value:
        return  # Suppress output if subsystem/loglevel is disabled

    #timestamp = time.strftime("%H:%M:%S.%f")[:-3]  # Format timestamp
    #timestamp = time.strftime("%H:%M:%S.%f")  # Format timestamp
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]	# Format timestamp

    lcolor = loglevel_colors[loglevel]
    scolor = subsystem_colors[subsystem]
    #print(f'{lcolor = }, {scolor = }')
    padded_loglevel  = colorize(loglevel.name.ljust (len(Subsystem.WARNING.name)),	ansi=lcolor.value)
    padded_subsystem = colorize(subsystem.name.ljust(len(Subsystem.WIDEST____.name)),	ansi=scolor.value)

    #Construct the output string
    #output_str = f"[{timestamp}][{lcolor}{padded_loglevel}{Style.RESET_ALL}] - [{scolor}{padded_subsystem}{Style.RESET_ALL}] "
    output_str = f"[{timestamp}][{padded_loglevel}] - [{padded_subsystem}] "

    #Handle f-strings and multiple arguments elegantly
    if len(args) == 1 and isinstance(args[0], str):
        output_str += args[0]
    else:
        output_str += sep.join(map(str, args))

    #print(f"{Style.BRIGHT}{Fore.WHITE}{output_str}{Style.RESET_ALL}", sep=sep, end=end, flush=flush)
    print(f"{output_str}", sep=sep, end=end, flush=flush)

def colorama_test_1():
	from colorama import Fore
	from colorama import init as colorama_init

	colorama_init(autoreset=True)

	colors = dict(Fore.__dict__.items())

	for color in colors.keys():
		print(colors[color] + f"{color}")

def colorama_test_2():
	from colorama import Fore
	from colorama import init as colorama_init
	from termcolor import colored
	
	colorama_init(autoreset=True)
	
	colors = [x for x in dir(Fore) if x[0] != "_"]
	colors = [i for i in colors if i not in ["BLACK", "RESET"] and "LIGHT" not in i] 
	
	for color  in colors:
		print(colored(color, color.lower()))


def dbgprint_test():
	colorama_test_1()
	colorama_test_2()

	# Example Usage
	dbgprint(threading, LogLevel.ERROR, f"Thread 123 exited with return value -1")
	dbgprint(queues, LogLevel.WARNING, "Queue is almost full", 95, "%")
	dbgprint(sharedmemory, LogLevel.DEBUG, "Shared memory segment allocated successfully")
	dbgprint(threading, LogLevel.TRACE, f"Thread 456 started") #This will not print because it's below the enabled loglevel for threading.
	
	# Example variables
	thread_id = 1
	retval = -1
	
	# Using f-string style
	dbgprint(threading, error, f"Thread {thread_id} exited with return value {retval}")
	
	# Using concatenated arguments
	dbgprint(threading, error, "Thread ", thread_id, " exited with return value ", retval)
	
	# Different loglevels and subsystems
	dbgprint(sharedmemory, trace, "Shared memory segment created")
	dbgprint(network, fatal, "Unrecoverable network error")
	dbgprint(queues, warning, "Queue overflow detected")

if __name__ == "__main__":
	dbgprint_test()

