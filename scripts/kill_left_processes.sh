kill $(ps aux | grep seg_lapa | grep -v grep | awk '{print $2}')
