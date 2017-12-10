import { Component } from '@angular/core';
import { MatDialog, MatDialogRef } from '@angular/material';
import { ProfileComponent } from './profile/profile.component';
import { Name, profiles } from './profile/profiles';


@Component({
  selector: 'app-footer',
  template: `
    <div class="footer-wrapper">
      <div class="credits">
        <button (click)="open(0)" mat-raised-button>Jan Haug</button>
        <button (click)="open(1)" mat-raised-button>Roman Schulte</button>
        <button (click)="open(2)" mat-raised-button>Marius Hobbhahn</button>
      </div>
    </div>
  `,
  styles: [`
    .footer-wrapper {
      width: 100%;
      height: 100%;
    }
  `]
})
export class FooterComponent {
  private _profileRef: MatDialogRef<ProfileComponent>;

  constructor(
    private _dialog: MatDialog
  ) { }

  public open(name: Name) {
    this._profileRef = this._dialog.open(
      ProfileComponent,
      {
        width: '300px',
        height: '600px'
      }
    );
    this._profileRef.componentInstance.profile = profiles[name];
  }
}

