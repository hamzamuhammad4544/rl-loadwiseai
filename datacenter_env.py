"""
Data-center flexible power management environment (Gymnasium).

Idea (inspired by Loadwise): a data center owns no generation equipment of its
own -- only a battery and some deferrable ("flexible") workload. It has two
separate incoming connections: the ordinary power grid (with a price, a
congestion state, and a connection limit) and a separate line to a solar
energy provider (e.g. a renewable power-purchase agreement) that supplies
cheap solar electricity whenever solar is being generated. Every hour the
data center decides how to use its battery and its flexible load so that it
pays as little as possible in total, stays under the grid connection limit
(tighter when the grid is congested), and does not let deferred jobs pile up
(SLA).

State  = (hour, battery level, price level, congestion flag, solar availability,
          deferred-job backlog)
Action = 0 hold | 1 charge battery | 2 discharge battery | 3 defer flexible load
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DataCenterEnv(gym.Env):
    # ---- sizes of each state feature -------------------------------------
    HOURS = 24
    N_SOC = 6        # battery level 0..5
    N_PRICE = 3      # 0 = low, 1 = medium, 2 = high
    N_CONG = 2       # 0 = normal, 1 = congested
    N_SOLAR = 3      # 0 = none, 1 = partial, 2 = full solar-provider availability
    N_BACKLOG = 3    # deferred jobs waiting: 0..2

    # ---- actions ----------------------------------------------------------
    HOLD, CHARGE, DISCHARGE, DEFER = 0, 1, 2, 3

    # ---- physical / economic parameters (all in "load units" per hour) ----
    BASE_LOAD = 2                        # IT + cooling demand
    SPIKE_PROB = 0.25                    # random extra demand of +1 unit
    NORMAL_LIMIT = 4                     # grid connection limit, normal hours
    CONG_LIMIT = 3                       # tighter limit when the grid is congested
    PRICE_EUR = np.array([0.2, 0.5, 1.2])  # cost per unit of grid energy
    SOLAR_RATE = 0.10                    # flat contracted cost per unit from the
                                          # solar-provider connection (cheaper than
                                          # even the cheapest grid rate, but not free)

    VIOLATION_PENALTY = 5.0   # per unit of grid draw above the limit
    OVERFLOW_PENALTY = 3.0    # deferring when the backlog is already full
    DELAY_COST = 0.3          # per waiting job per hour (SLA pressure)
    WEAR_COST = 0.05          # battery wear per charge/discharge
    END_BACKLOG_PENALTY = 3.0 # per unfinished job at end of day
    SALVAGE_PER_LEVEL = 0.5   # value of leftover battery energy at end of day

    # ---- solar schedule -----------------------------------------------
    # Hour ranges used to shape the solar-availability distribution. Night
    # hours never produce solar; "peak" hours (around solar noon) are most
    # likely to reach full output; "mid" and "dawn/dusk" hours taper off.
    NIGHT_HOURS = set(range(0, 6)) | set(range(19, 24))
    DAWN_DUSK_HOURS = {6, 7, 17, 18}
    MID_HOURS = {8, 9, 15, 16}
    PEAK_HOURS = {10, 11, 12, 13, 14}

    def __init__(self):
        super().__init__()
        self._dims = (self.HOURS, self.N_SOC, self.N_PRICE, self.N_CONG,
                      self.N_SOLAR, self.N_BACKLOG)
        self.nS = int(np.prod(self._dims))
        self.nA = 4
        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)
        self.state = None

    # ---- state <-> integer index -----------------------------------------
    def encode(self, hour, soc, price, cong, solar, backlog):
        return int(np.ravel_multi_index((hour, soc, price, cong, solar, backlog), self._dims))

    def decode(self, s):
        return tuple(int(x) for x in np.unravel_index(s, self._dims))

    # ---- exogenous (grid) dynamics ---------------------------------------
    @staticmethod
    def _price_dist(hour):
        if hour < 6:   return np.array([0.70, 0.25, 0.05])   # night: cheap
        if hour < 17:  return np.array([0.20, 0.60, 0.20])   # day: medium
        if hour < 22:  return np.array([0.05, 0.25, 0.70])   # evening peak: expensive
        return np.array([0.30, 0.50, 0.20])                  # late evening

    @staticmethod
    def _cong_prob(hour):
        return 0.35 if 17 <= hour < 22 else 0.08

    def _sample_grid(self, hour, price, cong):
        """Next-hour price and congestion (Markov, hour-dependent)."""
        if self.np_random.random() < 0.6:
            next_price = price                                   # price tends to persist
        else:
            next_price = int(self.np_random.choice(3, p=self._price_dist(hour)))
        p_cong = 0.5 * cong + 0.5 * self._cong_prob(hour)        # congestion persists too
        next_cong = int(self.np_random.random() < p_cong)
        return next_price, next_cong

    def _sample_solar(self, hour):
        """Solar-provider availability this hour: 0 (none), 1 (partial), 2 (full)
        units of cheap solar electricity available on the separate solar-provider
        connection. Deterministically zero at night; during the day it is random
        (cloud cover), with the chance of full output highest around solar noon."""
        if hour in self.NIGHT_HOURS:
            return 0
        if hour in self.PEAK_HOURS:
            levels, probs = [2, 1, 0], [0.65, 0.30, 0.05]
        elif hour in self.MID_HOURS:
            levels, probs = [2, 1, 0], [0.35, 0.50, 0.15]
        else:  # dawn / dusk: low sun angle, never reaches "full"
            levels, probs = [1, 0], [0.70, 0.30]
        return int(self.np_random.choice(levels, p=probs))

    # ---- Gymnasium API -----------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is not None and "state" in options:
            # Optional "exploring start": begin from any chosen state. Used by
            # the dynamics estimator so it can visit every state.
            self.state = self.decode(options["state"])
        else:
            soc = int(self.np_random.choice([3, 5]))             # half or full battery
            price = int(self.np_random.choice(3, p=self._price_dist(0)))
            cong = int(self.np_random.random() < self._cong_prob(0))
            solar = self._sample_solar(0)                        # hour 0 is night -> 0
            self.state = (0, soc, price, cong, solar, 0)          # start at midnight
        return self.encode(*self.state), {}

    def step(self, action):
        hour, soc, price, cong, solar, backlog = self.state

        # 1) demand this hour (random spike makes the reward stochastic)
        load = self.BASE_LOAD + int(self.np_random.random() < self.SPIKE_PROB)

        # 2) flexible-load handling
        sla_penalty = 0.0
        new_backlog = backlog
        if action == self.DEFER:
            load -= 1                                            # throttle one unit
            if backlog == self.N_BACKLOG - 1:
                sla_penalty += self.OVERFLOW_PENALTY             # queue full -> job dropped
            else:
                new_backlog += 1
        elif backlog > 0:
            load += 1                                            # catch up on one deferred job
            new_backlog -= 1

        # 3) battery handling
        raw_draw, new_soc, wear = load, soc, 0.0
        if action == self.CHARGE and soc < self.N_SOC - 1:
            raw_draw += 1; new_soc += 1; wear = self.WEAR_COST
        elif action == self.DISCHARGE and soc > 0:
            raw_draw -= 1; new_soc -= 1; wear = self.WEAR_COST
        raw_draw = max(raw_draw, 0)

        # 4) the separate solar-provider connection first serves whatever would
        #    otherwise be drawn (load and/or battery charging), up to however
        #    much solar is available this hour, at its own flat contracted
        #    rate. It is a different physical connection from the grid, so it
        #    does not count against the grid connection limit -- but it is not
        #    free. Anything beyond the solar connection's capacity is drawn
        #    from the grid at that hour's price and does count against the
        #    grid limit.
        solar_used = min(raw_draw, solar)
        grid_draw = max(0, raw_draw - solar)

        # 5) reward = - (energy cost + grid-limit violation + SLA + wear)
        limit = self.CONG_LIMIT if cong else self.NORMAL_LIMIT
        excess = max(0, grid_draw - limit)
        cost = self.SOLAR_RATE * solar_used + self.PRICE_EUR[price] * grid_draw
        delay = self.DELAY_COST * new_backlog
        reward = -(cost + self.VIOLATION_PENALTY * excess + sla_penalty + wear + delay)

        # 6) advance time
        next_hour = hour + 1
        terminated = next_hour == self.HOURS
        if terminated:
            reward += -self.END_BACKLOG_PENALTY * new_backlog + self.SALVAGE_PER_LEVEL * new_soc
            next_hour = self.HOURS - 1       # keep the observation a valid index
        next_price, next_cong = self._sample_grid(min(next_hour, self.HOURS - 1), price, cong)
        next_solar = self._sample_solar(min(next_hour, self.HOURS - 1))

        self.state = (next_hour, new_soc, next_price, next_cong, next_solar, new_backlog)
        info = {"raw_draw": raw_draw, "grid_draw": grid_draw, "solar_used": solar_used,
                "excess": excess, "cost": float(cost)}
        return self.encode(*self.state), float(reward), terminated, False, info
