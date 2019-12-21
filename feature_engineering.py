import numpy as np

def count_column(df, column):
    tp = df.groupby(column).count().reset_index()
    tp = tp[list(tp.columns)[0:2]]
    tp.columns = [column, column + '_count']
    df = df.merge(tp, on=column, how='left')
    return df


def get_statistical_fea(df, base_column, count_column):
    print("--------------------")
    group = df.groupby(base_column)
    tp = group.agg({count_column: ['mean']}).reset_index()
    tp.columns = [base_column, base_column + '_' + count_column + '_mean']
    df = df.merge(tp, on=base_column, how='left')

    tp = group.agg({count_column: ['count']}).reset_index()
    tp.columns = [base_column, base_column + '_' + count_column + '_count']
    df = df.merge(tp, on=base_column, how='left')

    tp = group.agg({count_column: ['sum']}).reset_index()
    tp.columns = [base_column, base_column + '_' + count_column + '_sum']
    df = df.merge(tp, on=base_column, how='left')

    tp = group.agg({count_column: ['std']}).reset_index()
    tp.columns = [base_column, base_column + '_' + count_column + '_std']
    df = df.merge(tp, on=base_column, how='left')
    return df


def count_count(df, base_column, count_column):
    tp = df.groupby(base_column).agg({count_column: ['count']}).reset_index()
    tp.columns = [base_column, base_column + '_' + count_column + '_count']
    df = df.merge(tp, on=base_column, how='left')
    return df


def count_sum(df, base_column, count_column):
    tp = df.groupby(base_column).agg({count_column: ['sum']}).reset_index()
    tp.columns = [base_column, base_column + '_' + count_column + '_sum']
    df = df.merge(tp, on=base_column, how='left')
    return df


def count_std(df, base_column, count_column):
    tp = df.groupby(base_column).agg({count_column: ['std']}).reset_index()
    tp.columns = [base_column, base_column + '_' + count_column + '_std']
    df = df.merge(tp, on=base_column, how='left')
    return df


def count_maxmin(df, base_column, count_column):
    tp_max = df.groupby(base_column).agg({count_column: np.max}).reset_index()
    tp_max.columns = [base_column, base_column + '_' + count_column + 'max']

    tp_min = df.groupby(base_column).agg({count_column: np.min}).reset_index()
    tp_min.columns = [base_column, base_column + '_' + count_column + 'min']

    tp_max = tp_max.merge(tp_min, on=base_column, how='left')
    tp_max[base_column + '_' + count_column + 'maxmin'] = (tp_max[base_column + '_' + count_column + 'max'] - tp_max[
        base_column + '_' + count_column + 'min']) / tp_max[base_column + '_' + count_column + 'max']
    tp = tp_max[[base_column, base_column + '_' + count_column + 'maxmin']]
    df = df.merge(tp, on=base_column, how='left')
    return df

def energy(df):
    x = df['jet_px']
    y = df['jet_py']
    z = df['jet_pz']
    return (x ** 2 + y ** 2 + z ** 2) ** 0.5

def calculate_feature(train, test):

    train['energy'] = train.apply(energy, axis=1)
    test['energy'] = test.apply(energy, axis=1)

    # add event_cnt
    # 区分单个和多个event的jet
    # train = count_column(train, 'event_id')
    # test = count_column(test, 'event_id')
    # 质能方程
    train['speed'] = (train['jet_energy'] / train['jet_mass']) ** 0.5
    test['speed'] = (test['jet_energy'] / test['jet_mass']) ** 0.5
    # 方向比例
    train['x/y'] = train['jet_px'] / train['jet_py']
    train['y/z'] = train['jet_py'] / train['jet_pz']
    train['z/x'] = train['jet_pz'] / train['jet_px']
    test['x/y'] = test['jet_px'] / test['jet_py']
    test['y/z'] = test['jet_py'] / test['jet_pz']
    test['z/x'] = test['jet_pz'] / test['jet_px']

    # 比例和
    train['xyz'] = train['x/y'] + train['y/z'] + train['z/x']
    test['xyz'] = test['x/y'] + test['y/z'] + test['z/x']

    train = get_statistical_fea(train, base_column=['event_id'], count_column=['xyz'])


    # event id 下 speed 统计特征
    # train = count_mean(train, 'event_id', 'speed')
    # train = count_sum(train, 'event_id', 'speed')
    # train = count_std(train, 'event_id', 'speed')
    # train = count_maxmin(train, 'event_id', 'speed')
    #
    # test = count_mean(test, 'event_id', 'speed')
    # test = count_sum(test, 'event_id', 'speed')
    # test = count_std(test, 'event_id', 'speed')
    # test = count_maxmin(test, 'event_id', 'speed')
    #
    # # event ID下 方向比例特征
    # train = count_mean(train, 'event_id', 'x/y')
    # train = count_sum(train, 'event_id', 'x/y')
    # train = count_std(train, 'event_id', 'x/y')
    # train = count_maxmin(train, 'event_id', 'x/y')
    #
    # test = count_mean(test, 'event_id', 'x/y')
    # test = count_sum(test, 'event_id', 'x/y')
    # test = count_std(test, 'event_id', 'x/y')
    # test = count_maxmin(test, 'event_id', 'x/y')
    #
    # train = count_mean(train, 'event_id', 'y/z')
    # train = count_sum(train, 'event_id', 'y/z')
    # train = count_std(train, 'event_id', 'y/z')
    # train = count_maxmin(train, 'event_id', 'y/z')
    #
    # test = count_mean(test, 'event_id', 'y/z')
    # test = count_sum(test, 'event_id', 'y/z')
    # test = count_std(test, 'event_id', 'y/z')
    # test = count_maxmin(test, 'event_id', 'y/z')
    #
    # train = count_mean(train, 'event_id', 'z/x')
    # train = count_sum(train, 'event_id', 'z/x')
    # train = count_std(train, 'event_id', 'z/x')
    # train = count_maxmin(train, 'event_id', 'z/x')
    #
    # test = count_mean(test, 'event_id', 'z/x')
    # test = count_sum(test, 'event_id', 'z/x')
    # test = count_std(test, 'event_id', 'z/x')
    # test = count_maxmin(test, 'event_id', 'z/x')
    #
    # # event id 下 比例和
    # train = count_mean(train, 'event_id', 'xyz')
    # train = count_sum(train, 'event_id', 'xyz')
    # train = count_std(train, 'event_id', 'xyz')
    # train = count_maxmin(train, 'event_id', 'xyz')
    #
    # test = count_mean(test, 'event_id', 'xyz')
    # test = count_sum(test, 'event_id', 'xyz')
    # test = count_std(test, 'event_id', 'xyz')
    # test = count_maxmin(test, 'event_id', 'xyz')
    #
    # # finish my add frist time
    #
    # # my 2nd add
    # train['speed_x'] = train['speed'] * train['jet_px']
    # train['speed_y'] = train['speed'] * train['jet_py']
    # train['speed_z'] = train['speed'] * train['jet_pz']
    #
    # test['speed_x'] = test['speed'] * test['jet_px']
    # test['speed_y'] = test['speed'] * test['jet_py']
    # test['speed_z'] = test['speed'] * test['jet_pz']
    #
    # # add tongji fea
    # train = count_mean(train, 'event_id', 'speed_x')
    # train = count_sum(train, 'event_id', 'speed_x')
    # train = count_std(train, 'event_id', 'speed_x')
    # train = count_maxmin(train, 'event_id', 'speed_x')
    #
    # train = count_mean(train, 'event_id', 'speed_y')
    # train = count_sum(train, 'event_id', 'speed_y')
    # train = count_std(train, 'event_id', 'speed_y')
    # train = count_maxmin(train, 'event_id', 'speed_y')
    #
    # train = count_mean(train, 'event_id', 'speed_z')
    # train = count_sum(train, 'event_id', 'speed_z')
    # train = count_std(train, 'event_id', 'speed_z')
    # train = count_maxmin(train, 'event_id', 'speed_z')
    #
    # test = count_mean(test, 'event_id', 'speed_x')
    # test = count_sum(test, 'event_id', 'speed_x')
    # test = count_std(test, 'event_id', 'speed_x')
    # test = count_maxmin(test, 'event_id', 'speed_x')
    #
    # test = count_mean(test, 'event_id', 'speed_y')
    # test = count_sum(test, 'event_id', 'speed_y')
    # test = count_std(test, 'event_id', 'speed_y')
    # test = count_maxmin(test, 'event_id', 'speed_y')
    #
    # test = count_mean(test, 'event_id', 'speed_z')
    # test = count_sum(test, 'event_id', 'speed_z')
    # test = count_std(test, 'event_id', 'speed_z')
    # test = count_maxmin(test, 'event_id', 'speed_z')
    # # finish my 2th add fea
    #
    # # my 3rd add feature
    # train['abs'] = train['jet_energy'] - train['energy']
    # test['abs'] = test['jet_energy'] - test['energy']
    #
    # train['mean_speed'] = train['speed'] / train['number_of_particles_in_this_jet']
    # test['mean_speed'] = test['speed'] / test['number_of_particles_in_this_jet']
    #
    # train['mean_ener'] = train['energy'] / train['number_of_particles_in_this_jet']
    # test['mean_ener'] = test['energy'] / test['number_of_particles_in_this_jet']
    #
    # train['mean_abs'] = train['abs'] / train['number_of_particles_in_this_jet']
    # test['mean_abs'] = test['abs'] / test['number_of_particles_in_this_jet']
    #
    # train = count_mean(train, 'event_id', 'mean_speed')
    # train = count_sum(train, 'event_id', 'mean_speed')
    # train = count_std(train, 'event_id', 'mean_speed')
    # train = count_maxmin(train, 'event_id', 'mean_speed')
    # test = count_mean(test, 'event_id', 'mean_speed')
    # test = count_sum(test, 'event_id', 'mean_speed')
    # test = count_std(test, 'event_id', 'mean_speed')
    # test = count_maxmin(test, 'event_id', 'mean_speed')
    #
    # train = count_mean(train, 'event_id', 'mean_ener')
    # train = count_sum(train, 'event_id', 'mean_ener')
    # train = count_std(train, 'event_id', 'mean_ener')
    # train = count_maxmin(train, 'event_id', 'mean_ener')
    # test = count_mean(test, 'event_id', 'mean_ener')
    # test = count_sum(test, 'event_id', 'mean_ener')
    # test = count_std(test, 'event_id', 'mean_ener')
    # test = count_maxmin(test, 'event_id', 'mean_ener')
    #
    # train = count_mean(train, 'event_id', 'mean_abs')
    # train = count_sum(train, 'event_id', 'mean_abs')
    # train = count_std(train, 'event_id', 'mean_abs')
    # train = count_maxmin(train, 'event_id', 'mean_abs')
    # test = count_mean(test, 'event_id', 'mean_abs')
    # test = count_sum(test, 'event_id', 'mean_abs')
    # test = count_std(test, 'event_id', 'mean_abs')
    # test = count_maxmin(test, 'event_id', 'mean_abs')
    #
    # # finish my 3rd add feature
    #
    # #my 4th add feature
    #
    # train['mean_speed_x'] = train['mean_speed'] * train['jet_px']
    # train['mean_speed_y'] = train['mean_speed'] * train['jet_py']
    # train['mean_speed_z'] = train['mean_speed'] * train['jet_pz']
    # test['mean_speed_x'] = test['mean_speed'] * test['jet_px']
    # test['mean_speed_y'] = test['mean_speed'] * test['jet_py']
    # test['mean_speed_z'] = test['mean_speed'] * test['jet_pz']
    #
    #
    # train = count_mean(train, 'event_id', 'mean_speed_x')
    # train = count_sum(train, 'event_id', 'mean_speed_x')
    # train = count_std(train, 'event_id', 'mean_speed_x')
    # train = count_maxmin(train, 'event_id', 'mean_speed_x')
    # test = count_mean(test, 'event_id', 'mean_speed_x')
    # test = count_sum(test, 'event_id', 'mean_speed_x')
    # test = count_std(test, 'event_id', 'mean_speed_x')
    # test = count_maxmin(test, 'event_id', 'mean_speed_x')
    #
    # train = count_mean(train, 'event_id', 'mean_speed_y')
    # train = count_sum(train, 'event_id', 'mean_speed_y')
    # train = count_std(train, 'event_id', 'mean_speed_y')
    # train = count_maxmin(train, 'event_id', 'mean_speed_y')
    # test = count_mean(test, 'event_id', 'mean_speed_y')
    # test = count_sum(test, 'event_id', 'mean_speed_y')
    # test = count_std(test, 'event_id', 'mean_speed_y')
    # test = count_maxmin(test, 'event_id', 'mean_speed_y')
    #
    # train = count_mean(train, 'event_id', 'mean_speed_z')
    # train = count_sum(train, 'event_id', 'mean_speed_z')
    # train = count_std(train, 'event_id', 'mean_speed_z')
    # train = count_maxmin(train, 'event_id', 'mean_speed_z')
    # test = count_mean(test, 'event_id', 'mean_speed_z')
    # test = count_sum(test, 'event_id', 'mean_speed_z')
    # test = count_std(test, 'event_id', 'mean_speed_z')
    # test = count_maxmin(test, 'event_id', 'mean_speed_z')
    #
    # #finish my 4th feature
    #
    # #my 5th feature about number
    # train = count_mean(train, 'number_of_particles_in_this_jet', 'speed')
    # train = count_sum(train, 'number_of_particles_in_this_jet', 'speed')
    # train = count_std(train, 'number_of_particles_in_this_jet', 'speed')
    # train = count_maxmin(train, 'number_of_particles_in_this_jet', 'speed')
    #
    # test = count_mean(test, 'number_of_particles_in_this_jet', 'speed')
    # test = count_sum(test, 'number_of_particles_in_this_jet', 'speed')
    # test = count_std(test, 'number_of_particles_in_this_jet', 'speed')
    # test = count_maxmin(test, 'number_of_particles_in_this_jet', 'speed')
    #
    # train = count_mean(train, 'number_of_particles_in_this_jet', 'energy')
    # train = count_sum(train, 'number_of_particles_in_this_jet', 'energy')
    # train = count_std(train, 'number_of_particles_in_this_jet', 'energy')
    # train = count_maxmin(train, 'number_of_particles_in_this_jet', 'energy')
    #
    # test = count_mean(test, 'number_of_particles_in_this_jet', 'energy')
    # test = count_sum(test, 'number_of_particles_in_this_jet', 'energy')
    # test = count_std(test, 'number_of_particles_in_this_jet', 'energy')
    # test = count_maxmin(test, 'number_of_particles_in_this_jet', 'energy')
    #
    # train = count_mean(train, 'number_of_particles_in_this_jet', 'jet_mass')
    # train = count_sum(train, 'number_of_particles_in_this_jet', 'jet_mass')
    # train = count_std(train, 'number_of_particles_in_this_jet', 'jet_mass')
    # train = count_maxmin(train, 'number_of_particles_in_this_jet', 'jet_mass')
    #
    # test = count_mean(test, 'number_of_particles_in_this_jet', 'jet_mass')
    # test = count_sum(test, 'number_of_particles_in_this_jet', 'jet_mass')
    # test = count_std(test, 'number_of_particles_in_this_jet', 'jet_mass')
    # test = count_maxmin(test, 'number_of_particles_in_this_jet', 'jet_mass')
    #
    # train = count_mean(train, 'number_of_particles_in_this_jet', 'jet_energy')
    # train = count_sum(train, 'number_of_particles_in_this_jet', 'jet_energy')
    # train = count_std(train, 'number_of_particles_in_this_jet', 'jet_energy')
    # train = count_maxmin(train, 'number_of_particles_in_this_jet', 'jet_energy')
    #
    # test = count_mean(test, 'number_of_particles_in_this_jet', 'jet_energy')
    # test = count_sum(test, 'number_of_particles_in_this_jet', 'jet_energy')
    # test = count_std(test, 'number_of_particles_in_this_jet', 'jet_energy')
    # test = count_maxmin(test, 'number_of_particles_in_this_jet', 'jet_energy')
    #
    # train = count_mean(train, 'number_of_particles_in_this_jet', 'abs')
    # train = count_sum(train, 'number_of_particles_in_this_jet', 'abs')
    # train = count_std(train, 'number_of_particles_in_this_jet', 'abs')
    # train = count_maxmin(train, 'number_of_particles_in_this_jet', 'abs')
    #
    # test = count_mean(test, 'number_of_particles_in_this_jet', 'abs')
    # test = count_sum(test, 'number_of_particles_in_this_jet', 'abs')
    # test = count_std(test, 'number_of_particles_in_this_jet', 'abs')
    # test = count_maxmin(test, 'number_of_particles_in_this_jet', 'abs')
    #
    #
    # train = count_count(train, 'number_of_particles_in_this_jet', 'event_id')
    # test = count_count(test, 'number_of_particles_in_this_jet', 'event_id')
    #
    # #finish my 5th feature
    # train['x_n'] = train['jet_px'] / train['energy']
    # train['y_n'] = train['jet_py'] / train['energy']
    # train['z_n'] = train['jet_pz'] / train['energy']
    #
    # test['x_n'] = test['jet_px'] / test['energy']
    # test['y_n'] = test['jet_py'] / test['energy']
    # test['z_n'] = test['jet_pz'] / test['energy']
    #
    # # begin 6th feature
    # train = count_mean(train, 'number_of_particles_in_this_jet', 'x_n')
    # train = count_sum(train, 'number_of_particles_in_this_jet', 'x_n')
    # train = count_std(train, 'number_of_particles_in_this_jet', 'x_n')
    # train = count_maxmin(train, 'number_of_particles_in_this_jet', 'x_n')
    #
    # test = count_mean(test, 'number_of_particles_in_this_jet', 'x_n')
    # test = count_sum(test, 'number_of_particles_in_this_jet', 'x_n')
    # test = count_std(test, 'number_of_particles_in_this_jet', 'x_n')
    # test = count_maxmin(test, 'number_of_particles_in_this_jet', 'x_n')
    #
    # train = count_mean(train, 'number_of_particles_in_this_jet', 'y_n')
    # train = count_sum(train, 'number_of_particles_in_this_jet', 'y_n')
    # train = count_std(train, 'number_of_particles_in_this_jet', 'y_n')
    # train = count_maxmin(train, 'number_of_particles_in_this_jet', 'y_n')
    #
    # test = count_mean(test, 'number_of_particles_in_this_jet', 'y_n')
    # test = count_sum(test, 'number_of_particles_in_this_jet', 'y_n')
    # test = count_std(test, 'number_of_particles_in_this_jet', 'y_n')
    # test = count_maxmin(test, 'number_of_particles_in_this_jet', 'y_n')
    #
    # train = count_mean(train, 'number_of_particles_in_this_jet', 'z_n')
    # train = count_sum(train, 'number_of_particles_in_this_jet', 'z_n')
    # train = count_std(train, 'number_of_particles_in_this_jet', 'z_n')
    # train = count_maxmin(train, 'number_of_particles_in_this_jet', 'z_n')
    #
    # test = count_mean(test, 'number_of_particles_in_this_jet', 'z_n')
    # test = count_sum(test, 'number_of_particles_in_this_jet', 'z_n')
    # test = count_std(test, 'number_of_particles_in_this_jet', 'z_n')
    # test = count_maxmin(test, 'number_of_particles_in_this_jet', 'z_n')
    #
    # # finish 6th feature
    #
    # train = count_mean(train, 'event_id', 'x_n')
    # train = count_sum(train, 'event_id', 'x_n')
    # train = count_std(train, 'event_id', 'x_n')
    # train = count_maxmin(train, 'event_id', 'x_n')
    #
    # train = count_mean(train, 'event_id', 'y_n')
    # train = count_sum(train, 'event_id', 'y_n')
    # train = count_std(train, 'event_id', 'y_n')
    # train = count_maxmin(train, 'event_id', 'y_n')
    #
    # train = count_mean(train, 'event_id', 'z_n')
    # train = count_sum(train, 'event_id', 'z_n')
    # train = count_std(train, 'event_id', 'z_n')
    # train = count_maxmin(train, 'event_id', 'z_n')
    #
    # test = count_mean(test, 'event_id', 'x_n')
    # test = count_sum(test, 'event_id', 'x_n')
    # test = count_std(test, 'event_id', 'x_n')
    # test = count_maxmin(test, 'event_id', 'x_n')
    #
    # test = count_mean(test, 'event_id', 'y_n')
    # test = count_sum(test, 'event_id', 'y_n')
    # test = count_std(test, 'event_id', 'y_n')
    # test = count_maxmin(test, 'event_id', 'y_n')
    #
    # test = count_mean(test, 'event_id', 'z_n')
    # test = count_sum(test, 'event_id', 'z_n')
    # test = count_std(test, 'event_id', 'z_n')
    # test = count_maxmin(test, 'event_id', 'z_n')
    #
    #
    #
    # train = count_mean(train, 'event_id', 'number_of_particles_in_this_jet')
    # train = count_sum(train, 'event_id', 'number_of_particles_in_this_jet')
    # train = count_std(train, 'event_id', 'number_of_particles_in_this_jet')
    # train = count_maxmin(train, 'event_id', 'number_of_particles_in_this_jet')
    #
    # train = count_mean(train, 'event_id', 'jet_mass')
    # train = count_sum(train, 'event_id', 'jet_mass')
    # train = count_std(train, 'event_id', 'jet_mass')
    # train = count_maxmin(train, 'event_id', 'jet_mass')
    #
    # train = count_mean(train, 'event_id', 'jet_energy')
    # train = count_sum(train, 'event_id', 'jet_energy')
    # train = count_std(train, 'event_id', 'jet_energy')
    # train = count_maxmin(train, 'event_id', 'jet_energy')
    #
    # train['mean_energy'] = train['jet_energy'] / train['number_of_particles_in_this_jet']
    # train['mean_jet_mass'] = train['jet_mass'] / train['number_of_particles_in_this_jet']
    #
    # train = count_mean(train, 'event_id', 'mean_energy')
    # train = count_sum(train, 'event_id', 'mean_energy')
    # train = count_std(train, 'event_id', 'mean_energy')
    # train = count_maxmin(train, 'event_id', 'mean_energy')
    #
    # train = count_mean(train, 'event_id', 'mean_jet_mass')
    # train = count_sum(train, 'event_id', 'mean_jet_mass')
    # train = count_std(train, 'event_id', 'mean_jet_mass')
    # train = count_maxmin(train, 'event_id', 'mean_jet_mass')
    #
    # train = count_mean(train, 'event_id', 'abs')
    # train = count_sum(train, 'event_id', 'abs')
    # train = count_std(train, 'event_id', 'abs')
    # train = count_maxmin(train, 'event_id', 'abs')
    #
    # train = count_mean(train, 'event_id', 'energy')
    # train = count_sum(train, 'event_id', 'energy')
    # train = count_std(train, 'event_id', 'energy')
    # train = count_maxmin(train, 'event_id', 'energy')
    #
    # test = count_mean(test, 'event_id', 'number_of_particles_in_this_jet')
    # test = count_sum(test, 'event_id', 'number_of_particles_in_this_jet')
    # test = count_std(test, 'event_id', 'number_of_particles_in_this_jet')
    # test = count_maxmin(test, 'event_id', 'number_of_particles_in_this_jet')
    #
    # test = count_mean(test, 'event_id', 'jet_mass')
    # test = count_sum(test, 'event_id', 'jet_mass')
    # test = count_std(test, 'event_id', 'jet_mass')
    # test = count_maxmin(test, 'event_id', 'jet_mass')
    #
    # test = count_mean(test, 'event_id', 'jet_energy')
    # test = count_sum(test, 'event_id', 'jet_energy')
    # test = count_std(test, 'event_id', 'jet_energy')
    # test = count_maxmin(test, 'event_id', 'jet_energy')
    #
    # test['mean_energy'] = test['jet_energy'] / test['number_of_particles_in_this_jet']
    # test['mean_jet_mass'] = test['jet_mass'] / test['number_of_particles_in_this_jet']
    #
    # test = count_mean(test, 'event_id', 'mean_energy')
    # test = count_sum(test, 'event_id', 'mean_energy')
    # test = count_std(test, 'event_id', 'mean_energy')
    # test = count_maxmin(test, 'event_id', 'mean_energy')
    #
    # test = count_mean(test, 'event_id', 'mean_jet_mass')
    # test = count_sum(test, 'event_id', 'mean_jet_mass')
    # test = count_std(test, 'event_id', 'mean_jet_mass')
    # test = count_maxmin(test, 'event_id', 'mean_jet_mass')
    #
    # test = count_mean(test, 'event_id', 'abs')
    # test = count_sum(test, 'event_id', 'abs')
    # test = count_std(test, 'event_id', 'abs')
    # test = count_maxmin(test, 'event_id', 'abs')
    #
    # test = count_mean(test, 'event_id', 'energy')
    # test = count_sum(test, 'event_id', 'energy')
    # test = count_std(test, 'event_id', 'energy')
    # test = count_maxmin(test, 'event_id', 'energy')
    #
    # print('finish feature making \nfeatures are:',train.columns)
    #
    # return train, test