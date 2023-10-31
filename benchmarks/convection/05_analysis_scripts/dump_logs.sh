#!/bin/bash

for e in `ls | grep run`; do
	 cp $e/log.out logs/$e.log
done
