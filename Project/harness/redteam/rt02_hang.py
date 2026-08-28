HYPOTHESIS = "red-team: must be killed by the per-iteration timeout"
def run(splits):
    import time; time.sleep(60)
    return {"valid": [], "test": []}
