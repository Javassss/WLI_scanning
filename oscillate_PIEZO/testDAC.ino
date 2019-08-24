// MCPDAC relies on SPI.
#include <SPI.h>
#include "MCPDAC.h"

void setup()
{
  // CS on pin 10, no LDAC pin (tie it to ground).
  Serial.begin(9600);
  MCPDAC.begin(10);
  Serial.setTimeout(20);
  // Set the gain to "HIGH" mode - 0 to 4096mV.
  MCPDAC.setGain(CHANNEL_A,GAIN_HIGH);

  // Do not shut down channel A, but shut down channel B.
  MCPDAC.shutdown(CHANNEL_A, false);
  MCPDAC.shutdown(CHANNEL_B, true);

  //MCPDAC.setVoltage(CHANNEL_A, 0&0x0fff);
}

void loop()
{
  static unsigned int volts = 0;
  static int incr = 10;
  Serial.println(volts);

  if (volts == 0) {
    MCPDAC.setVoltage(CHANNEL_A, volts & 0x0fff);
    //delay(5000);
    volts += incr;
  } else {
    MCPDAC.setVoltage(CHANNEL_A, volts & 0x0fff);
    delay(20);
    // Increase the voltage
    volts += incr;

    if (volts >= 4096)
    {
      volts = 0;
    }
  }

}
