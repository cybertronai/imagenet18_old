#!/usr/bin/env python
#
# Prepares DAWN TSV file from TensorBoard events url

import sys, os, re
from dateutil import parser

events_url = 'https://s3.amazonaws.com/yaroslavvb/logs/release-sixteen.04.events'

import os
import glob
import numpy as np
import datetime as dt
import pytz
from tensorflow.python.summary import summary_iterator
import argparse

parser = argparse.ArgumentParser(description='launch')
parser.add_argument('--ignore-eval', action='store_true', 
                    help='ignore eval time')
args = parser.parse_args()

def get_events(fname, x_axis='step'):
  """Returns event dictionary for given run, has form
  {tag1: {step1: val1}, tag2: ..}

  If x_axis is set to "time", step is replaced by timestamp
  """
  result = {}
  
  events = summary_iterator.summary_iterator(fname)

  try:
    for event in events:
      if x_axis == 'step':
        x_val = event.step
      elif x_axis == 'time':
        x_val = event.wall_time
      else:
        assert False, f"Unknown x_axis ({x_axis})"

      vals = {val.tag: val.simple_value for val in event.summary.value}
      # step_time: value
      for tag in vals:
        event_dict = result.setdefault(tag, {})
        if x_val in event_dict:
          print(f"Warning, overwriting {tag} for {x_axis}={x_val}")
          print(f"old val={event_dict[x_val]}")
          print(f"new val={vals[tag]}")

        event_dict[x_val] = vals[tag]
  except Exception as e:
    print(e)
    pass
        
  return result

def datetime_from_seconds(seconds, timezone="US/Pacific"):
  """
  timezone: pytz timezone name to use for conversion, ie, UTC or US/Pacific
  """
  return dt.datetime.fromtimestamp(seconds, pytz.timezone(timezone))


def download_file(url):
  import urllib.request
  response = urllib.request.urlopen(url)
  data = response.read()    
  return data

def main():
  with open('/tmp/events', 'wb') as f:
    f.write(download_file(events_url))


  events_dict=get_events('/tmp/events', 'step')
  
  # build step->time dict for eval events
  lr = events_dict['sizes/lr']
  for step in lr:
    print('{"learning_rate": '+str(lr[step])+', "example": '+str(step)+"},")

if __name__=='__main__':
  main()
