
#### Signal

Сигнал производиться, когда какой-то стейт изменяется. Signals are public access functions and can be emitted from anywhere, but we recommend to only emit them from the class that defines the signal and its subclasses.

Когда сигнал испускается, то его слот <-> функция выполняется сразу же, просто как function call. 

Если несколько слотов подключено к одному сигналу, то они будут запускаться (слоты) в порядке очереди присоединения.

Важно, что слоты не должны, что то возвращать

### Slots

A slot called when a signal connected to it is emitted. Они простые функции и просто выполняются единственный прикол в том, что они могу подключиться к сигналу.

Если смотреть C++, то `using slot = std::function<void(Args...)>; ` 

### A C++ implementation

