#!/bin/bash
# watch the most recent log file in ./log
watch tail `ls -1td ./log/*| head -n1`  
