import pandas as pd, numpy as np
from yj import Shaper, FManager
from yj.runners import BaseRunnerImpl
import lightgbm as lgb
import gc
from datetime import timedelta

class LGBMRunner(BaseRunnerImpl.BaseRunnerImpl):

    def _predict(self, prices, calendar, trainCols, model):
        from yj.environment import ALPHAS, WEIGHTS, TR_LAST, MAX_LAGS, FDAY, SUBMISSION_FP

        # PREDICTIONS
        self.lg.debug("making predictions...")

        for icount, (alpha, weight) in enumerate(zip(ALPHAS, WEIGHTS)):
            te = Shaper.create_ds(TR_LAST, MAX_LAGS, calendar, prices)
            cols = [f"F{i}" for i in range(1, 29)]

            for tdelta in range(0, 28):
                day = FDAY + timedelta(days=tdelta)
                self.lg.debug("%s, %s" % (tdelta, day))
                tst = te[(te['date'] >= day - timedelta(days=MAX_LAGS)) & (te['date'] <= day)].copy()
                Shaper.create_features(tst)
                tst = tst.loc[tst['date'] == day, trainCols]
                te.loc[te['date'] == day, "sales"] = alpha * model.predict(tst)  # magic multiplier by kyakovlev

            te_sub = te.loc[te['date'] >= FDAY, ["id", "sales"]].copy()
            te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount() + 1]
            te_sub = te_sub.set_index(["id", "F"]).unstack()["sales"][cols].reset_index()
            te_sub.fillna(0., inplace=True)
            te_sub.sort_values("id", inplace=True)
            te_sub.reset_index(drop=True, inplace=True)
            te_sub.to_csv(f"submission_{icount}.csv", index=False)

            if icount == 0:
                sub = te_sub
                sub[cols] *= weight
            else:
                sub[cols] += te_sub[cols] * weight

            self.lg.debug(icount, alpha, weight)

        self.lg.debug("creating csv file submissions")
        sub2 = sub.copy()
        sub2["id"] = sub2["id"].str.replace("validation", "evaluation")
        sub = pd.concat([sub, sub2], axis=0, sort=False)
        sub.to_csv(SUBMISSION_FP, index=False)

    def train(self, cuda):

        from yj.environment import LGBM_PARAMS, LGBM_MODEL_FP, CAT_FEATURES, X_Y_FPS

        # define categorical features
        X_train, y_train = FManager.load_objs(X_Y_FPS)

        validInds = np.random.choice(X_train.index.values, 2_000_000, replace=False)
        trainInds = np.setdiff1d(X_train.index.values, validInds)

        trainData = lgb.Dataset(X_train.loc[trainInds], label=y_train.loc[trainInds],
                                categorical_feature=CAT_FEATURES, free_raw_data=False)
        validData = lgb.Dataset(X_train.loc[validInds], label=y_train.loc[validInds],
                                categorical_feature=CAT_FEATURES, free_raw_data=False)

        self.lg.debug("cleaning up necessarily data structures...")
        del X_train, y_train, validInds, trainInds
        gc.collect()

        self.lg.debug("training lgb model...")

        # Train LightGBM model
        m_lgb = lgb.train(LGBM_PARAMS, trainData, valid_sets=[validData], verbose_eval=20)

        self.lg.debug("saving the model...")
        # # Save the model
        m_lgb.save_model(LGBM_MODEL_FP)
        self.lg.debug("model saved... done.")

    def predict(self):
        from yj.environment import LGBM_MODEL_FP, PRICE_CAL_TRAIN_DF_FP

        trainCols, calendar, prices = FManager.load_objs(PRICE_CAL_TRAIN_DF_FP)
        m_lgb = lgb.Booster(model_file=LGBM_MODEL_FP)
        self._predict(prices, calendar, trainCols, m_lgb)