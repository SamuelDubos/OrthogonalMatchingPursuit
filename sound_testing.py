from matplotlib.pyplot import figure, plot, show
from numpy import frombuffer, sign
from wave import open


def get_signal(filename):

    wave = open(filename, 'r')
    buffer = wave.readframes(-1)
    signal = frombuffer(buffer, 'int16')

    return signal


def clip_signal(filename, threshold):

    signal = get_signal(filename)
    output = []
    for value in signal:
        if abs(value) > threshold:
            output.append(sign(value) * threshold)
        else:
            output.append(value)

    return output


if __name__ == '__main__':

    plot_all_sounds = False
    plot_clipped_bassoon = True

    SOURCE_PATH = 'C://Users//samue//declipping2020_codes//Sounds'
    INSTRUMENTS = ['VIOLIN', 'CLARINET', 'BASSOON', 'HARP', 'GLOCKENSPIEL',
                   'CELESTA', 'ACCORDION', 'GUITAR', 'PIANO', 'WIND']
    NAMES = ['a08_violin', 'a16_clarinet', 'a18_bassoon', 'a25_harp',
             'a35_glockenspiel', 'a41_celesta', 'a42_accordion',
             'a60_piano_schubert', 'a66_wind_ensemble_stravinsky']
    FILENAMES = [f'{SOURCE_PATH}//{NAME}.wav' for NAME in NAMES]
    SOUNDS = dict(zip(INSTRUMENTS, FILENAMES))

    if plot_all_sounds:

        for INSTRUMENT, FILENAME in SOUNDS.items():
            figure(INSTRUMENT)
            plot(get_signal(FILENAME))

    if plot_clipped_bassoon:

        figure('Normal')
        plot(get_signal(SOUNDS['BASSOON']))
        figure('Clipped')
        plot(clip_signal(SOUNDS['BASSOON'], threshold=1e4))

    show()
