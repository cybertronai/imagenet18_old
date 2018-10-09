#!/usr/bin/env python
#
# Prepares DAWN TSV file from TensorBoard events url

events_url = 'https://s3.amazonaws.com/yaroslavvb/logs/release-sixteen.04.events'

import datetime as dt
import pytz
from tensorflow.python.summary import summary_iterator
import argparse

parser = argparse.ArgumentParser(description='launch')
parser.add_argument('--ignore-eval', action='store_true',
                    help='ignore eval time')
args = parser.parse_args()


def get_events(fname, x_axis='step'):
    """Returns event dictionary for given run.

    Has form {tag1: {step1: val1}, tag2: ..}

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
    """Timezone: pytz timezone name to use for conversion, ie, UTC or US/Pacific."""
    return dt.datetime.fromtimestamp(seconds, pytz.timezone(timezone))


def download_file(url):
    import urllib.request
    response = urllib.request.urlopen(url)
    data = response.read()
    return data


def main():
    with open('/tmp/events', 'wb') as f:
        f.write(download_file(events_url))

    events_dict = get_events('/tmp/events', 'step')
    events_dict2 = get_events('/tmp/events', 'time')
    # starting time, "first" event gets logged in beginning of main()
    first = events_dict2['first']
    start_time = list(first.keys())[0]

    # build step->time dict for eval events
    events_step = events_dict['losses/test_5']
    # steps = list(events_step.keys())
    events_time = events_dict2['losses/test_5']
    # times = list(events_time.keys())
    step_time = {v[0]: v[1] for v in zip(events_step, events_time)}
    print(step_time)

    # get ending time
    test_5 = events_dict['losses/test_5']
    test_1 = events_dict['losses/test_1']
    eval_sec = events_dict['times/eval_sec']
    total_eval_sec = 0
    for (i, step) in enumerate(test_1):
        # subtract eval time, which is not required
        # https://github.com/stanford-futuredata/dawn-bench-entries/issues/12#issuecomment-381363792
        ts = step_time[step]
        elapsed = ts - start_time
        if args.ignore_eval:
            total_eval_sec += eval_sec[step]
            elapsed -= total_eval_sec

        print(f"{i+1}\t{(elapsed/3600)}\t{test_1[step]}\t{test_5[step]}")
        if test_5[step] >= 93:
            # end_time = ts
            break


if __name__ == '__main__':
    main()
