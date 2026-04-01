# Pipeline Monitoring Commands

## Current Run (parallel, 3 workers)
Main PID: 599996
Workers: 602552, 602553, 602554
Started: 2026-02-09 ~16:32
Log: /home/oneknight/projects/bangla-llm/data-pipeline/pipeline_run.log

## Check worker status (all 3 should show ~100% CPU)
```
ps -p 602552,602553,602554 -o pid,stat,etime,pcpu,rss --no-headers
```

## Check if main process is alive
```
ps -p 599996 > /dev/null 2>&1 && echo "ALIVE" || echo "DEAD"
```

## Stream log (updates only after all chunks finish processing)
```
tail -f /home/oneknight/projects/bangla-llm/data-pipeline/pipeline_run.log
```

## Memory check (should have >5GB available)
```
free -h | head -2
```

## Expected timeline
- Processing phase: ~4-5 hours (3 workers, ~311K docs each)
- Dedup phase: ~30 min (serial merge)
- Total: ~5 hours

## After completion
```
cat /home/oneknight/projects/bangla-llm/data-pipeline/data/reports/quality_report.json | python -m json.tool
```

## To kill if needed
```
kill 599996; sleep 2; pkill -f "pipeline_parallel"
```
