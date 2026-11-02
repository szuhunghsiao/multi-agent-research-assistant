import asyncio

from app.jobs import JobManager


class SlowGraph:
    async def ainvoke(self, state):
        await asyncio.sleep(10)


class InnerTimeoutGraph:
    """A call inside the graph timed out (e.g. Resilient gave up) -- not the job deadline."""
    async def ainvoke(self, state):
        raise TimeoutError("researcher call timed out")


async def run_one(graph, job_timeout_s):
    m = JobManager(graph, workers=1, job_timeout_s=job_timeout_s)
    m.start()
    job = m.submit("topic")
    await m.queue.join()
    await m.stop()
    return m.get(job.job_id)


async def test_job_deadline_marks_failed_and_frees_worker():
    job = await run_one(SlowGraph(), job_timeout_s=0.1)
    assert job.status == "failed"
    assert job.error == "JobTimeout: exceeded 0.1s"


async def test_inner_timeout_is_not_reported_as_job_deadline():
    job = await run_one(InnerTimeoutGraph(), job_timeout_s=5)
    assert job.status == "failed"
    assert job.error == "TimeoutError: researcher call timed out"