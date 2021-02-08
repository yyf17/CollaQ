import datetime
import os
import pprint
import time

def get_game_name(args):
    if 'map_name' in args.env_args:
        map_name = ""
        for map1 in args.env_args['map_name']:
            map_name = f"{map_name}-{map1}"
        unique_token = "{}__{}__{}".format(
            args.name,
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            map_name
        )
        #AttributeError: 'list' object has no attribute 'replace'
        game_name =  map_name

    else:
        unique_token = "{}__{}__{}".format(
            args.name,
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            args.env
        )
        game_name = args.env

    return game_name,unique_token

